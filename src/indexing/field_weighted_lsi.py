"""
Field-weighted LSI implementation.
Extends the basic LSI with separate field vectors and adaptive field weighting.
"""

import os
import numpy as np
import joblib
import logging
from typing import Dict, List, Any, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from scipy.sparse import spmatrix, diags, vstack

from .base import BaseIndexer
from .lsi import LSIIndexer
from .utils import combine_fields

class FieldWeightedLSIIndexer(LSIIndexer):
    """Field-weighted LSI indexer implementation with adaptive field weighting."""
    
    def __init__(self, n_components: int = 150):
        """
        Initialize the field-weighted LSI indexer.
        
        Args:
            n_components: Number of latent semantic dimensions
        """
        super().__init__(n_components)
        self.index_name = "lsi_field_weighted"
        self.vectorizer = None  # Single vectorizer for all fields
        self.svd = None  # Single SVD model
        self._field_vectors = {}  # TF-IDF vectors for each field
        self._field_available = {}  # Track which fields are available
        self._field_weights = {}  # Field weights
    
    def build_index(self, documents: Dict[str, List], output_dir: str) -> None:
        """
        Build a field-weighted LSI index from documents.
        
        Args:
            documents: Dictionary containing document fields
            output_dir: Directory to save the index
        """
        logging.info(f"Building field-weighted LSI index with {self.n_components} dimensions")
        
        # Store document info for search results
        self._store_document_info(documents)
        
        # Step 1: Combine all fields to create a common vocabulary
        logging.info("Creating common vocabulary from all fields")
        combined_texts = combine_fields(documents)
        
        # Step 2: Create and fit the TF-IDF vectorizer on combined texts
        logging.info("Fitting TF-IDF vectorizer on combined texts")
        self.vectorizer = self._create_tfidf_vectorizer()
        self.vectorizer.fit(combined_texts)
        
        # Step 3: Create TF-IDF vectors for each field separately
        logging.info("Creating separate TF-IDF vectors for each field")
        self._create_field_vectors(documents)
        
        # Step 4: Apply adaptive field weighting
        logging.info("Applying adaptive field weighting")
        combined_vectors = self._apply_field_weighting()
        
        # Step 5: Apply LSI to the combined weighted vectors
        logging.info(f"Applying SVD to reduce to {self.n_components} dimensions")
        self._doc_vectors = self._apply_lsi(combined_vectors)
        
        # Save the index
        self._save_index(output_dir)
        logging.info("Field-weighted LSI index built successfully")
    
    def _create_field_vectors(self, documents: Dict[str, List]) -> None:
        """
        Create TF-IDF vectors for each field using the common vocabulary.
        
        Args:
            documents: Dictionary containing document fields
        """
        # For each available field, create TF-IDF vectors
        self._field_vectors = {}
        self._field_available = {}
        num_docs = len(documents['paper_ids'])
        
        for field_name in self._available_field_names:
            if field_name in documents and any(documents[field_name]):
                logging.info(f"Processing field: {field_name}")
                
                # For fields with missing values, replace with empty strings
                field_texts = []
                field_mask = np.zeros(num_docs)
                
                for i in range(num_docs):
                    if i < len(documents[field_name]) and documents[field_name][i]:
                        field_texts.append(documents[field_name][i])
                        field_mask[i] = 1.0
                    else:
                        field_texts.append("")
                
                # Transform field texts using the common vectorizer
                try:
                    field_vectors = self.vectorizer.transform(field_texts)
                    
                    # Store the field vectors and availability mask
                    self._field_vectors[field_name] = field_vectors
                    self._field_available[field_name] = field_mask
                    
                    logging.info(f"Field {field_name}: {field_vectors.shape[1]} terms, {np.sum(field_mask)} non-empty documents")
                except Exception as e:
                    logging.warning(f"Error processing field {field_name}: {e}")
    
    def _apply_field_weighting(self) -> spmatrix:
        """
        Apply field weights to TF-IDF vectors with adaptive scaling for missing fields.
        
        Returns:
            Combined weighted document vectors
        """
        # Get field weights from default weights
        self._field_weights = {
            field_name.rstrip('s'): self.DEFAULT_FIELD_WEIGHTS.get(field_name.rstrip('s'), 1.0)
            for field_name in self._field_vectors.keys()
        }
        
        # Calculate the total weight (theoretical maximum if all fields present)
        total_weight = sum(self._field_weights.values())
        
        # Initialize the combined vectors
        num_docs = len(self._doc_info)
        combined_vectors = None
        
        # Create masks for empty documents in each field
        field_masks = {}
        doc_weights = np.zeros(num_docs)
        
        # Process each field
        for field_name in self._field_vectors.keys():
            field_key = field_name.rstrip('s')
            field_weight = self._field_weights.get(field_key, 1.0)
            
            # Get the field availability mask
            field_mask = self._field_available.get(field_name, np.zeros(num_docs))
            field_masks[field_name] = field_mask
            
            # Add the weight to the document weights where field is available
            doc_weights += field_mask * field_weight
        
        # Calculate adaptive scaling factors
        scaling_factors = np.divide(total_weight, doc_weights, 
                                   out=np.ones_like(doc_weights), 
                                   where=doc_weights>0)
        
        # Apply weights with scaling to each field
        for field_name, field_vectors in self._field_vectors.items():
            field_key = field_name.rstrip('s')
            field_weight = self._field_weights.get(field_key, 1.0)
            field_mask = field_masks[field_name]
            
            # Create scaled weights for this field
            scaled_weights = field_mask * field_weight * scaling_factors
            
            # Apply the weights to the field vectors using diagonal matrix
            weighted_vectors = diags(scaled_weights).dot(field_vectors)
            
            # Add to combined vectors
            if combined_vectors is None:
                combined_vectors = weighted_vectors
            else:
                combined_vectors += weighted_vectors
        
        # Store the scaling factors for reference
        self._scaling_factors = scaling_factors.tolist()
        
        return combined_vectors
    
    def _apply_lsi(self, X_tfidf: spmatrix) -> np.ndarray:
        """
        Apply LSI (truncated SVD) to a TF-IDF matrix.
        
        Args:
            X_tfidf: TF-IDF matrix
            
        Returns:
            Document vectors in LSI space
        """
        n_components = min(self.n_components, X_tfidf.shape[1] - 1, X_tfidf.shape[0] - 1)
        if n_components <= 0:
            raise ValueError("Not enough documents/terms for SVD")
            
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        doc_vectors = self.svd.fit_transform(X_tfidf)
        
        # Normalize document vectors
        return normalize(doc_vectors)
    
    def _save_index(self, output_dir: str) -> None:
        """
        Save the field-weighted LSI index to disk.
        
        Args:
            output_dir: Directory to save the index
        """
        logging.info(f"Saving field-weighted LSI index to {output_dir}")
        
        # Create metadata
        self._metadata = {
            'index_type': self.index_name,
            'num_documents': len(self._doc_info),
            'n_components': self.n_components,
            'fields': list(self._field_vectors.keys()),
            'field_weights': self._field_weights,
            'explained_variance': np.sum(self.svd.explained_variance_ratio_) if self.svd is not None else 0,
            'field_scaling': {
                'total_weight': sum(self._field_weights.values()),
                'adaptive_scaling_applied': True
            }
        }
        
        # Create directories if they don't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save vectorizer (common for all fields)
        joblib.dump(self.vectorizer, os.path.join(output_dir, f'{self.index_name}_vectorizer.joblib'))
        
        # Save SVD model
        joblib.dump(self.svd, os.path.join(output_dir, f'{self.index_name}_svd.joblib'))
        
        # Save field vectors
        for field, vectors in self._field_vectors.items():
            joblib.dump(vectors, os.path.join(output_dir, f'{self.index_name}_{field}_vectors.joblib'))
        
        # Save field availability masks
        joblib.dump(self._field_available, os.path.join(output_dir, f'{self.index_name}_field_available.joblib'))
        
        # Save document vectors
        joblib.dump(self._doc_vectors, os.path.join(output_dir, f'{self.index_name}_doc_vectors.joblib'))
        
        # Save document info
        joblib.dump(self._doc_info, os.path.join(output_dir, f'{self.index_name}_doc_info.joblib'))
        
        # Save metadata
        self._save_metadata(output_dir)
    
    def load_index(self, index_dir: str) -> None:
        """
        Load a field-weighted LSI index from disk.
        
        Args:
            index_dir: Directory containing the index
        """
        logging.info(f"Loading field-weighted LSI index from {index_dir}")
        
        # Load metadata first to get fields
        self._load_metadata(index_dir)
        
        # Set parameters from metadata
        if 'n_components' in self._metadata:
            self.n_components = self._metadata['n_components']
        
        if 'field_weights' in self._metadata:
            self._field_weights = self._metadata['field_weights']
        
        # Load vectorizer (common for all fields)
        self.vectorizer = joblib.load(os.path.join(index_dir, f'{self.index_name}_vectorizer.joblib'))
        
        # Load SVD model
        self.svd = joblib.load(os.path.join(index_dir, f'{self.index_name}_svd.joblib'))
        
        # Load field availability masks
        field_available_path = os.path.join(index_dir, f'{self.index_name}_field_available.joblib')
        if os.path.exists(field_available_path):
            self._field_available = joblib.load(field_available_path)
        
        # Load field vectors
        self._field_vectors = {}
        if 'fields' in self._metadata:
            for field in self._metadata['fields']:
                vectors_path = os.path.join(index_dir, f'{self.index_name}_{field}_vectors.joblib')
                if os.path.exists(vectors_path):
                    self._field_vectors[field] = joblib.load(vectors_path)
        
        # Load document vectors
        self._doc_vectors = joblib.load(os.path.join(index_dir, f'{self.index_name}_doc_vectors.joblib'))
        
        # Load document info
        self._doc_info = joblib.load(os.path.join(index_dir, f'{self.index_name}_doc_info.joblib'))
        
        logging.info(f"{self.index_name} index loaded successfully")
    
    def _process_query(self, query: str) -> np.ndarray:
        """
        Process a query string into a field-weighted LSI vector.
        
        Args:
            query: Search query string
            
        Returns:
            Query vector in LSI space
        """
        if self.vectorizer is None or self.svd is None:
            raise ValueError("Index must be built or loaded before searching")
            
        # Transform query using the common vectorizer
        query_tfidf = self.vectorizer.transform([query])
        
        # Project query into LSI space
        query_lsi = self.svd.transform(query_tfidf)
        
        # Normalize query vector
        return normalize(query_lsi)
    
    def search(self, query: str, top_k: int = 50) -> List[Dict[str, Any]]:
        """
        Search for documents using field-weighted LSI similarity.
        
        Args:
            query: Search query string
            top_k: Number of results to return
            
        Returns:
            List of search results with scores
        """
        if self._doc_vectors is None:
            raise ValueError("Index must be built or loaded before searching")
        
        # Process query
        query_vector = self._process_query(query)
        
        # Calculate cosine similarity
        scores = np.dot(self._doc_vectors, query_vector.T).flatten()
        
        # Get top-k results
        return self._get_top_results(scores, top_k) 
"""
Field-weighted LSI implementation.
Extends the basic LSI with separate field vectors and combined weighting.
"""

import os
import numpy as np
import joblib
import logging
from typing import Dict, List, Any, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from scipy.sparse import spmatrix

from .base import BaseIndexer
from .lsi import LSIIndexer

class FieldWeightedLSIIndexer(LSIIndexer):
    """Field-weighted LSI indexer implementation."""
    
    def __init__(self, n_components: int = 150):
        """
        Initialize the field-weighted LSI indexer.
        
        Args:
            n_components: Number of latent semantic dimensions
        """
        super().__init__(n_components)
        self.index_name = "lsi_field_weighted"
        self.field_vectorizers = {}
        self.field_svds = {}
        self._field_vectors = {}
        self._available_fields = ['titles', 'abstracts', 'bodies', 'topics', 'keywords']
    
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
        
        # Process each field separately
        self._process_fields(documents)
        
        # Combine the field vectors with weights
        self._combine_field_vectors()
        
        # Save the index
        self._save_index(output_dir)
        logging.info("Field-weighted LSI index built successfully")
    
    def _process_fields(self, documents: Dict[str, List]) -> None:
        """
        Process each field separately.
        
        Args:
            documents: Dictionary containing document fields
        """
        for field in self._available_fields:
            if field in documents and any(documents[field]):
                logging.info(f"Processing field: {field}")
                
                # For fields with missing values, replace with empty strings
                field_texts = [text if text else "" for text in documents[field]]
                
                # Create TF-IDF matrix for this field using the parent's method
                field_matrix, vectorizer, _ = self._create_tfidf_matrix(field_texts)
                
                # Store the vectorizer
                self.field_vectorizers[field] = vectorizer
                
                # Apply LSI to this field using the parent's method
                try:
                    field_vectors = self._apply_lsi_to_field(field_matrix)
                    
                    # Store the field vectors
                    self._field_vectors[field] = field_vectors
                    
                    logging.info(f"Field {field}: {field_vectors.shape[1]} dimensions")
                except ValueError as e:
                    logging.warning(f"Error processing field {field}: {e}")
    
    def _apply_lsi_to_field(self, field_matrix: spmatrix) -> np.ndarray:
        """
        Apply LSI to a field TF-IDF matrix.
        
        Args:
            field_matrix: Field TF-IDF matrix
            
        Returns:
            Field vectors in LSI space
        """
        n_components = min(self.n_components, field_matrix.shape[1] - 1, field_matrix.shape[0] - 1)
        if n_components <= 0:
            raise ValueError("Not enough documents/terms for SVD")
            
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        field_vectors = svd.fit_transform(field_matrix)
        
        # Store the SVD model
        field_name = next((field for field, vec in self.field_vectorizers.items() 
                          if id(vec) == id(self.field_vectorizers[field])), None)
        if field_name:
            self.field_svds[field_name] = svd
        
        # Normalize the field vectors
        return normalize(field_vectors)
    
    def _combine_field_vectors(self) -> None:
        """Combine field vectors using the field weights."""
        n_docs = len(self._doc_info)
        n_dims = self.n_components
        
        # Initialize combined vectors
        combined_vectors = np.zeros((n_docs, n_dims))
        
        # Track weights used for each document
        doc_weights = np.zeros(n_docs)
        
        # Combine field vectors based on weights
        for field, vectors in self._field_vectors.items():
            # Skip fields with no vectors
            if vectors.shape[0] == 0:
                continue
                
            # Get the weight for this field
            field_weight = self.DEFAULT_FIELD_WEIGHTS.get(field.rstrip('s'), 1.0)
            
            # Pad or trim vectors to match n_dims
            padded_vectors = np.zeros((vectors.shape[0], n_dims))
            padded_vectors[:, :vectors.shape[1]] = vectors
            
            # Add weighted vectors to combined vectors
            combined_vectors += padded_vectors * field_weight
            doc_weights += field_weight
        
        # Normalize by total weight for each document
        doc_weights = np.maximum(doc_weights, 1e-10)  # Avoid division by zero
        for i in range(n_docs):
            combined_vectors[i] /= doc_weights[i]
        
        # Normalize the combined vectors
        self._doc_vectors = normalize(combined_vectors)
    
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
            'field_weights': self.DEFAULT_FIELD_WEIGHTS
        }
        
        # Create directories if they don't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save field vectorizers
        for field, vectorizer in self.field_vectorizers.items():
            joblib.dump(vectorizer, os.path.join(output_dir, f'{self.index_name}_{field}_vectorizer.joblib'))
        
        # Save field SVD models
        for field, svd in self.field_svds.items():
            joblib.dump(svd, os.path.join(output_dir, f'{self.index_name}_{field}_svd.joblib'))
        
        # Save field vectors
        for field, vectors in self._field_vectors.items():
            joblib.dump(vectors, os.path.join(output_dir, f'{self.index_name}_{field}_vectors.joblib'))
        
        # Save combined document vectors
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
        
        # Load field vectorizers, SVD models, and vectors
        if 'fields' in self._metadata:
            for field in self._metadata['fields']:
                vectorizer_path = os.path.join(index_dir, f'{self.index_name}_{field}_vectorizer.joblib')
                svd_path = os.path.join(index_dir, f'{self.index_name}_{field}_svd.joblib')
                vectors_path = os.path.join(index_dir, f'{self.index_name}_{field}_vectors.joblib')
                
                if os.path.exists(vectorizer_path):
                    self.field_vectorizers[field] = joblib.load(vectorizer_path)
                
                if os.path.exists(svd_path):
                    self.field_svds[field] = joblib.load(svd_path)
                
                if os.path.exists(vectors_path):
                    self._field_vectors[field] = joblib.load(vectors_path)
        
        # Load combined document vectors
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
        # Process query for each field
        field_query_vectors = {}
        for field, vectorizer in self.field_vectorizers.items():
            if field in self.field_svds:
                # Transform query to TF-IDF vector for this field
                query_tfidf = vectorizer.transform([query])
                
                # Project query into LSI space for this field
                svd = self.field_svds[field]
                query_lsi = svd.transform(query_tfidf)
                
                # Store query vector for this field
                field_query_vectors[field] = query_lsi
        
        # Combine query vectors with weights
        combined_query = np.zeros((1, self.n_components))
        total_weight = 0
        
        for field, query_vector in field_query_vectors.items():
            field_weight = self.DEFAULT_FIELD_WEIGHTS.get(field.rstrip('s'), 1.0)
            
            # Pad or trim vector to match n_dims
            padded_vector = np.zeros((1, self.n_components))
            padded_vector[:, :query_vector.shape[1]] = query_vector
            
            # Add weighted vector to combined vector
            combined_query += padded_vector * field_weight
            total_weight += field_weight
        
        # Normalize by total weight
        if total_weight > 0:
            combined_query /= total_weight
        
        # Normalize query vector
        return normalize(combined_query)
    
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search for documents using field-weighted LSI similarity.
        
        Args:
            query: Search query string
            top_k: Number of results to return
            
        Returns:
            List of search results with scores
        """
        if not self.field_vectorizers or self._doc_vectors is None:
            raise ValueError("Index must be built or loaded before searching")
        
        # Process query
        query_vector = self._process_query(query)
        
        # Calculate cosine similarity between query and documents
        scores = np.dot(self._doc_vectors, query_vector.T).flatten()
        
        # Get top-k results
        return self._get_top_results(scores, top_k) 
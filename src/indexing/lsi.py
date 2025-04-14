"""
Basic LSI (Latent Semantic Indexing) implementation.
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
from .utils import combine_fields

class LSIIndexer(BaseIndexer):
    """Basic LSI indexer implementation without field weighting."""
    
    def __init__(self, n_components: int = 150):
        """
        Initialize the LSI indexer.
        
        Args:
            n_components: Number of latent semantic dimensions
        """
        super().__init__("lsi_basic")
        self.n_components = n_components
        self.vectorizer = None
        self.svd = None
        self._doc_vectors = None
        self._doc_info = []
        self._terms = []
    
    def build_index(self, documents: Dict[str, List], output_dir: str) -> None:
        """
        Build an LSI index from documents.
        
        Args:
            documents: Dictionary containing document fields
            output_dir: Directory to save the index
        """
        logging.info(f"Building LSI index with {self.n_components} dimensions")
        
        # Store document info for search results
        self._store_document_info(documents)
        
        # Combine fields without weights - simple concatenation
        logging.info("Combining document fields without weighting")
        combined_texts = combine_fields(documents)
        
        # Create TF-IDF matrix
        X_tfidf, self.vectorizer, self._terms = self._create_tfidf_matrix(combined_texts)
        
        # Apply LSI
        self._doc_vectors = self._apply_lsi(X_tfidf)
        
        # Save the index
        self._save_index(output_dir)
        logging.info("LSI index built successfully")
    
    def _store_document_info(self, documents: Dict[str, List]) -> None:
        """
        Store document information for search results.
        
        Args:
            documents: Dictionary containing document fields
        """
        self._doc_info = []
        for i in range(len(documents['paper_ids'])):
            self._doc_info.append({
                'paper_id': documents['paper_ids'][i],
                'title': documents['titles'][i],
                'abstract': documents['abstracts'][i]
            })
    
    def _create_tfidf_vectorizer(self) -> TfidfVectorizer:
        """
        Create and configure a TF-IDF vectorizer.
        
        Returns:
            Configured TF-IDF vectorizer
        """
        return TfidfVectorizer(
            min_df=2,                  # Ignore terms in fewer than 2 documents
            max_df=0.95,               # Ignore terms in more than 95% of documents
            stop_words='english',      # Remove English stop words
            lowercase=True,            # Convert all text to lowercase
            strip_accents='unicode',   # Remove accents
            token_pattern=r'\b[a-zA-Z]{3,}\b',  # Only include words with 3+ letters
            norm='l2',                 # L2 normalization
            use_idf=True,              # Use inverse document frequency
            smooth_idf=True            # Add 1 to document frequencies to avoid division by zero
        )
    
    def _create_tfidf_matrix(self, texts: List[str]) -> Tuple[spmatrix, TfidfVectorizer, List[str]]:
        """
        Create TF-IDF matrix from texts.
        
        Args:
            texts: List of document texts
            
        Returns:
            Tuple of (TF-IDF matrix, vectorizer, terms)
        """
        logging.info("Creating TF-IDF document-term matrix")
        vectorizer = self._create_tfidf_vectorizer()
        X_tfidf = vectorizer.fit_transform(texts)
        terms = vectorizer.get_feature_names_out()
        return X_tfidf, vectorizer, terms
    
    def _apply_lsi(self, X_tfidf: spmatrix) -> np.ndarray:
        """
        Apply LSI (truncated SVD) to a TF-IDF matrix.
        
        Args:
            X_tfidf: TF-IDF matrix
            
        Returns:
            Document vectors in LSI space
        """
        logging.info(f"Applying SVD to reduce to {self.n_components} dimensions")
        n_components = min(self.n_components, X_tfidf.shape[1] - 1, X_tfidf.shape[0] - 1)
        if n_components <= 0:
            raise ValueError("Not enough documents/terms for SVD")
            
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        doc_vectors = self.svd.fit_transform(X_tfidf)
        
        # Normalize document vectors
        return normalize(doc_vectors)
    
    def _save_index(self, output_dir: str) -> None:
        """
        Save the LSI index to disk.
        
        Args:
            output_dir: Directory to save the index
        """
        logging.info(f"Saving LSI index to {output_dir}")
        
        # Create metadata
        self._metadata = {
            'index_type': self.index_name,
            'num_documents': len(self._doc_info),
            'num_terms': len(self._terms),
            'n_components': self.n_components,
            'explained_variance': np.sum(self.svd.explained_variance_ratio_) if self.svd is not None else 0
        }
        
        # Create directories if they don't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save vectorizer
        joblib.dump(self.vectorizer, os.path.join(output_dir, f'{self.index_name}_vectorizer.joblib'))
        
        # Save SVD model
        joblib.dump(self.svd, os.path.join(output_dir, f'{self.index_name}_svd.joblib'))
        
        # Save document vectors
        joblib.dump(self._doc_vectors, os.path.join(output_dir, f'{self.index_name}_doc_vectors.joblib'))
        
        # Save terms
        joblib.dump(self._terms, os.path.join(output_dir, f'{self.index_name}_terms.joblib'))
        
        # Save document info
        joblib.dump(self._doc_info, os.path.join(output_dir, f'{self.index_name}_doc_info.joblib'))
        
        # Save metadata
        self._save_metadata(output_dir)
    
    def load_index(self, index_dir: str) -> None:
        """
        Load an LSI index from disk.
        
        Args:
            index_dir: Directory containing the index
        """
        logging.info(f"Loading {self.index_name} index from {index_dir}")
        
        # Load vectorizer
        self.vectorizer = joblib.load(os.path.join(index_dir, f'{self.index_name}_vectorizer.joblib'))
        
        # Load SVD model
        self.svd = joblib.load(os.path.join(index_dir, f'{self.index_name}_svd.joblib'))
        
        # Load document vectors
        self._doc_vectors = joblib.load(os.path.join(index_dir, f'{self.index_name}_doc_vectors.joblib'))
        
        # Load terms
        self._terms = joblib.load(os.path.join(index_dir, f'{self.index_name}_terms.joblib'))
        
        # Load document info
        self._doc_info = joblib.load(os.path.join(index_dir, f'{self.index_name}_doc_info.joblib'))
        
        # Load metadata
        self._load_metadata(index_dir)
        
        # Set parameters from metadata if available
        if 'n_components' in self._metadata:
            self.n_components = self._metadata['n_components']
        
        logging.info(f"{self.index_name} index loaded successfully")
    
    def _process_query(self, query: str) -> np.ndarray:
        """
        Process a query string into an LSI vector.
        
        Args:
            query: Search query string
            
        Returns:
            Query vector in LSI space
        """
        if self.vectorizer is None or self.svd is None:
            raise ValueError("Index must be built or loaded before searching")
            
        # Transform query to TF-IDF vector
        query_tfidf = self.vectorizer.transform([query])
        
        # Project query into LSI space
        query_lsi = self.svd.transform(query_tfidf)
        
        # Normalize query vector
        return normalize(query_lsi)
    
    def search(self, query: str, top_k: int = 50) -> List[Dict[str, Any]]:
        """
        Search for documents using LSI similarity.
        
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
        
        # Calculate cosine similarity between query and documents
        # (dot product of normalized vectors = cosine similarity)
        scores = np.dot(self._doc_vectors, query_vector.T).flatten()
        
        # Get top-k results
        return self._get_top_results(scores, top_k)
    
    def _get_top_results(self, scores: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        """
        Get top-k results from scores.
        
        Args:
            scores: Document similarity scores
            top_k: Number of results to return
            
        Returns:
            List of search results with scores
        """
        top_indices = np.argsort(-scores)[:top_k]
        
        results = []
        for i, idx in enumerate(top_indices):
            if scores[idx] > 0:
                result = {
                    'score': float(scores[idx]),
                    'paper_id': self._doc_info[idx]['paper_id'],
                    'title': self._doc_info[idx]['title'],
                    'abstract': self._doc_info[idx]['abstract'],
                    'rank': i + 1
                }
                results.append(result)
        
        return results 
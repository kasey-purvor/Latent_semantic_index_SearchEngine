"""
BM25 Indexer implementation.
"""

import os
import numpy as np
import joblib
import logging
from typing import Dict, List, Any, Tuple
from sklearn.feature_extraction.text import CountVectorizer

from .base import BaseIndexer
from .utils import combine_fields

class BM25Indexer(BaseIndexer):
    """BM25 indexer implementation."""
    
    def __init__(self):
        """Initialize the BM25 indexer."""
        super().__init__("bm25")
        self.vectorizer = None
        self._doc_vectors = None
        self._doc_lengths = None
        self._avg_doc_length = 0
        self._idf = None
        self._doc_info = []
        
        # BM25 parameters
        self.k1 = 1.5  # Controls term frequency scaling
        self.b = 0.75  # Controls document length normalization
    
    def build_index(self, documents: Dict[str, List], output_dir: str) -> None:
        """
        Build a BM25 index from documents.
        
        Args:
            documents: Dictionary containing document fields
            output_dir: Directory to save the index
        """
        logging.info("Building BM25 index")
        
        # Store document info for search results
        self._doc_info = []
        for i in range(len(documents['paper_ids'])):
            self._doc_info.append({
                'paper_id': documents['paper_ids'][i],
                'title': documents['titles'][i],
                'abstract': documents['abstracts'][i]
            })
        
        # Combine fields without weights
        logging.info("Combining document fields")
        combined_texts = combine_fields(documents)
        
        # Create vectorizer
        logging.info("Creating document-term matrix")
        self.vectorizer = CountVectorizer(
            min_df=2,                  # Ignore terms in fewer than 2 documents
            max_df=0.95,               # Ignore terms in more than 95% of documents
            stop_words='english',      # Remove English stop words
            lowercase=True,            # Convert all text to lowercase
            strip_accents='unicode',   # Remove accents
            token_pattern=r'\b[a-zA-Z]{3,}\b'  # Only include words with 3+ letters
        )
        
        # Create document-term matrix
        self._doc_vectors = self.vectorizer.fit_transform(combined_texts)
        
        # Calculate document lengths
        self._doc_lengths = np.array(self._doc_vectors.sum(axis=1)).flatten()
        self._avg_doc_length = np.mean(self._doc_lengths)
        
        # Calculate IDF values
        n_docs = self._doc_vectors.shape[0]
        df = np.array(self._doc_vectors.sum(axis=0)).flatten()
        # Add 1 to DF to avoid division by zero
        df = np.where(df == 0, 1, df)
        # BM25 IDF formula: log((N - n + 0.5) / (n + 0.5))
        self._idf = np.log((n_docs - df + 0.5) / (df + 0.5) + 1)
        
        # Save the index
        self._save_index(output_dir)
        logging.info("BM25 index built successfully")
    
    def _save_index(self, output_dir: str) -> None:
        """
        Save the BM25 index to disk.
        
        Args:
            output_dir: Directory to save the index
        """
        logging.info(f"Saving BM25 index to {output_dir}")
        
        # Create metadata
        self._metadata = {
            'index_type': self.index_name,
            'num_documents': len(self._doc_info),
            'num_terms': self._doc_vectors.shape[1] if self._doc_vectors is not None else 0,
            'avg_doc_length': self._avg_doc_length,
            'k1': self.k1,
            'b': self.b
        }
        
        # Create directories if they don't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save vectorizer
        joblib.dump(self.vectorizer, os.path.join(output_dir, f'{self.index_name}_vectorizer.joblib'))
        
        # Save document vectors
        joblib.dump(self._doc_vectors, os.path.join(output_dir, f'{self.index_name}_doc_vectors.joblib'))
        
        # Save document lengths
        joblib.dump(self._doc_lengths, os.path.join(output_dir, f'{self.index_name}_doc_lengths.joblib'))
        
        # Save IDF values
        joblib.dump(self._idf, os.path.join(output_dir, f'{self.index_name}_idf.joblib'))
        
        # Save document info
        joblib.dump(self._doc_info, os.path.join(output_dir, f'{self.index_name}_doc_info.joblib'))
        
        # Save metadata
        self._save_metadata(output_dir)
    
    def load_index(self, index_dir: str) -> None:
        """
        Load a BM25 index from disk.
        
        Args:
            index_dir: Directory containing the index
        """
        logging.info(f"Loading BM25 index from {index_dir}")
        
        # Load vectorizer
        self.vectorizer = joblib.load(os.path.join(index_dir, f'{self.index_name}_vectorizer.joblib'))
        
        # Load document vectors
        self._doc_vectors = joblib.load(os.path.join(index_dir, f'{self.index_name}_doc_vectors.joblib'))
        
        # Load document lengths
        self._doc_lengths = joblib.load(os.path.join(index_dir, f'{self.index_name}_doc_lengths.joblib'))
        
        # Load IDF values
        self._idf = joblib.load(os.path.join(index_dir, f'{self.index_name}_idf.joblib'))
        
        # Load document info
        self._doc_info = joblib.load(os.path.join(index_dir, f'{self.index_name}_doc_info.joblib'))
        
        # Load metadata
        self._load_metadata(index_dir)
        
        # Set parameters from metadata if available
        if 'k1' in self._metadata:
            self.k1 = self._metadata['k1']
        if 'b' in self._metadata:
            self.b = self._metadata['b']
        if 'avg_doc_length' in self._metadata:
            self._avg_doc_length = self._metadata['avg_doc_length']
        
        logging.info("BM25 index loaded successfully")
    
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search for documents using BM25 ranking.
        
        Args:
            query: Search query string
            top_k: Number of results to return
            
        Returns:
            List of search results with scores
        """
        if self.vectorizer is None or self._doc_vectors is None:
            raise ValueError("Index must be built or loaded before searching")
        
        # Transform query to term vector
        query_vector = self.vectorizer.transform([query])
        
        # Get non-zero terms in the query
        query_terms = query_vector.indices
        
        # Calculate BM25 scores
        scores = np.zeros(self._doc_vectors.shape[0])
        
        # Process each query term separately to avoid memory issues
        for term_idx in query_terms:
            # Get the column for this term (document frequencies)
            term_vector = self._doc_vectors.getcol(term_idx).toarray().flatten()
            
            # Term frequency in each document
            tf = term_vector
            
            # Term frequency normalized by document length
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (self._doc_lengths / self._avg_doc_length))
            
            # IDF for this term
            idf = self._idf[term_idx]
            
            # Add the BM25 score for this term to the total scores
            scores += idf * (numerator / denominator)
        
        # Get top-k results
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
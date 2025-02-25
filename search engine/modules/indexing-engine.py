import os
import logging
import pickle
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from tqdm import tqdm
import h5py
import joblib
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("IndexingEngine")

class IndexingEngine:
    """
    Creates and manages the LSI index for academic papers.
    
    This engine handles the following tasks:
    1. Creating field-weighted TF-IDF matrices
    2. Applying truncated SVD for dimensionality reduction
    3. Storing document vectors efficiently
    
    It's designed to work with batches of documents to handle large datasets
    on machines with limited memory.
    """
    
    def __init__(self, 
                 index_dir: str,
                 n_components: int = 150,
                 field_weights: Dict[str, float] = None,
                 min_df: int = 5,
                 max_df: float = 0.8,
                 use_idf: bool = True,
                 chunk_size: int = 10000):
        """
        Initialize the indexing engine.
        
        Args:
            index_dir: Directory to store index files
            n_components: Number of dimensions to keep in SVD
            field_weights: Dictionary mapping field names to weights
            min_df: Minimum document frequency for terms
            max_df: Maximum document frequency for terms (float between 0.0 and 1.0)
            use_idf: Whether to use inverse document frequency weighting
            chunk_size: Number of documents to process at once for incremental SVD
        """
        self.index_dir = Path(index_dir)
        self.n_components = n_components
        self.field_weights = field_weights or {'title': 3.0, 'abstract': 1.5, 'full_text': 1.0}
        self.min_df = min_df
        self.max_df = max_df
        self.use_idf = use_idf
        self.chunk_size = chunk_size
        
        # Create vectorizers for each field
        self.vectorizers = {}
        
        # Create SVD model
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        
        # Document mapping: doc_id -> index in matrix
        self.doc_id_to_idx = {}
        # Term mapping: term -> index in matrix
        self.term_to_idx = {}
        
        # Ensure index directory exists
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"IndexingEngine initialized with {n_components} LSI components")
        logger.info(f"Field weights: {self.field_weights}")
        
    def _initialize_vectorizers(self):
        """Initialize TF-IDF vectorizers for each field."""
        for field in self.field_weights.keys():
            self.vectorizers[field] = TfidfVectorizer(
                min_df=self.min_df,
                max_df=self.max_df,
                use_idf=self.use_idf,
                norm='l2',
                analyzer='word',
                stop_words='english',
                token_pattern=r'\b[a-zA-Z][a-zA-Z0-9]*\b'
            )
        logger.info(f"Initialized vectorizers for fields: {list(self.field_weights.keys())}")
    
    def fit_vectorizers(self, documents: List[Dict[str, Any]]):
        """
        Fit the TF-IDF vectorizers on a collection of documents.
        
        Args:
            documents: List of processed documents
        """
        self._initialize_vectorizers()
        
        # Prepare text collections for each field
        field_texts = {field: [] for field in self.field_weights.keys()}
        
        for doc in documents:
            for field in self.field_weights.keys():
                if field in doc and doc[field]:
                    field_texts[field].append(doc[field])
                else:
                    field_texts[field].append("")
        
        # Fit each vectorizer
        for field, texts in field_texts.items():
            logger.info(f"Fitting vectorizer for field: {field} with {len(texts)} documents")
            self.vectorizers[field].fit(texts)
            
            # Update term mapping
            vocabulary = self.vectorizers[field].vocabulary_
            for term, idx in vocabulary.items():
                if term not in self.term_to_idx:
                    self.term_to_idx[term] = len(self.term_to_idx)
        
        logger.info(f"Vectorizers fitted. Total unique terms: {len(self.term_to_idx)}")
    
    def _create_weighted_matrix(self, documents: List[Dict[str, Any]]) -> sp.csr_matrix:
        """
        Create a weighted TF-IDF matrix from a batch of documents.
        
        Args:
            documents: List of processed documents
            
        Returns:
            Sparse matrix with weighted TF-IDF values
        """
        # Map documents to indices
        for doc in documents:
            if doc['id'] not in self.doc_id_to_idx:
                self.doc_id_to_idx[doc['id']] = len(self.doc_id_to_idx)
        
        # Get the shape of the final matrix
        n_docs = len(self.doc_id_to_idx)
        n_terms = len(self.term_to_idx)
        
        # Create a list to hold matrices for each field
        field_matrices = []
        
        # Process each field
        for field, weight in self.field_weights.items():
            if field not in self.vectorizers:
                continue
                
            # Extract texts for this field
            field_texts = []
            doc_indices = []
            
            for i, doc in enumerate(documents):
                if field in doc and doc[field]:
                    field_texts.append(doc[field])
                    doc_indices.append(self.doc_id_to_idx[doc['id']])
            
            if not field_texts:
                continue
                
            # Transform texts to TF-IDF matrix
            field_matrix = self.vectorizers[field].transform(field_texts)
            
            # Apply field weight
            field_matrix = field_matrix * weight
            
            # Create a properly sized matrix for all documents
            full_field_matrix = sp.lil_matrix((n_docs, field_matrix.shape[1]))
            
            # Fill in the values for documents that have this field
            for i, doc_idx in enumerate(doc_indices):
                full_field_matrix[doc_idx] = field_matrix[i]
            
            field_matrices.append(full_field_matrix.tocsr())
        
        # Stack matrices horizontally if we have more than one field
        if len(field_matrices) > 1:
            combined_matrix = sp.hstack(field_matrices).tocsr()
        elif len(field_matrices) == 1:
            combined_matrix = field_matrices[0]
        else:
            # Create an empty matrix if no fields were processed
            combined_matrix = sp.csr_matrix((n_docs, 0))
            
        return combined_matrix
    
    def fit_transform(self, documents: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit TF-IDF vectorizers and transform documents to LSI space.
        
        Args:
            documents: List of processed documents
            
        Returns:
            Tuple of (document vectors, term vectors)
        """
        # Fit vectorizers if not already fitted
        if not self.vectorizers:
            self.fit_vectorizers(documents)
        
        # Create weighted TF-IDF matrix
        weighted_matrix = self._create_weighted_matrix(documents)
        
        # Apply SVD
        logger.info(f"Applying SVD to matrix of shape {weighted_matrix.shape} with {self.n_components} components")
        document_vectors = self.svd.fit_transform(weighted_matrix)
        
        # Get term vectors (V matrix in SVD)
        term_vectors = self.svd.components_.T
        
        logger.info(f"SVD complete. Document vectors shape: {document_vectors.shape}")
        
        return document_vectors, term_vectors
    
    def transform(self, documents: List[Dict[str, Any]]) -> np.ndarray:
        """
        Transform documents to LSI space using fitted vectorizers and SVD.
        
        Args:
            documents: List of processed documents
            
        Returns:
            Document vectors in LSI space
        """
        # Create weighted TF-IDF matrix
        weighted_matrix = self._create_weighted_matrix(documents)
        
        # Apply SVD
        document_vectors = self.svd.transform(weighted_matrix)
        
        return document_vectors
    
    def save_index(self):
        """Save the index components to disk."""
        index_path = self.index_dir / "lsi_index"
        
        # Save vectorizers
        with open(index_path.with_suffix('.vectorizers'), 'wb') as f:
            pickle.dump(self.vectorizers, f)
        
        # Save SVD model
        joblib.dump(self.svd, index_path.with_suffix('.svd'))
        
        # Save document mapping
        with open(index_path.with_suffix('.doc_map'), 'wb') as f:
            pickle.dump(self.doc_id_to_idx, f)
        
        # Save term mapping
        with open(index_path.with_suffix('.term_map'), 'wb') as f:
            pickle.dump(self.term_to_idx, f)
        
        logger.info(f"Index saved to {index_path}")
    
    def load_index(self):
        """Load the index components from disk."""
        index_path = self.index_dir / "lsi_index"
        
        # Load vectorizers
        with open(index_path.with_suffix('.vectorizers'), 'rb') as f:
            self.vectorizers = pickle.load(f)
        
        # Load SVD model
        self.svd = joblib.load(index_path.with_suffix('.svd'))
        
        # Load document mapping
        with open(index_path.with_suffix('.doc_map'), 'rb') as f:
            self.doc_id_to_idx = pickle.load(f)
        
        # Load term mapping
        with open(index_path.with_suffix('.term_map'), 'rb') as f:
            self.term_to_idx = pickle.load(f)
        
        logger.info(f"Index loaded from {index_path}")
        logger.info(f"Index contains {len(self.doc_id_to_idx)} documents and {len(self.term_to_idx)} terms")
    
    def save_document_vectors(self, document_vectors: np.ndarray, doc_ids: List[str]):
        """
        Save document vectors to disk efficiently.
        
        Args:
            document_vectors: Matrix of document vectors
            doc_ids: List of document IDs corresponding to the vectors
        """
        vectors_path = self.index_dir / "document_vectors.h5"
        
        # Open HDF5 file in append mode
        with h5py.File(vectors_path, 'a') as f:
            # Create dataset if it doesn't exist
            if 'vectors' not in f:
                max_docs = len(self.doc_id_to_idx) * 2  # Allow for growth
                f.create_dataset('vectors', 
                                shape=(max_docs, self.n_components),
                                maxshape=(None, self.n_components),
                                dtype='float32',
                                chunks=(min(1000, max_docs), self.n_components))
                                
                # Create a dataset for document IDs
                dt = h5py.special_dtype(vlen=str)
                f.create_dataset('doc_ids', 
                                shape=(max_docs,),
                                maxshape=(None,),
                                dtype=dt)
            
            # Get current size
            current_size = f['vectors'].shape[0]
            
            # Resize if necessary
            if current_size < len(document_vectors):
                f['vectors'].resize((len(document_vectors), self.n_components))
                f['doc_ids'].resize((len(document_vectors),))
            
            # Store vectors and IDs
            for i, (doc_id, vector) in enumerate(zip(doc_ids, document_vectors)):
                doc_idx = self.doc_id_to_idx[doc_id]
                f['vectors'][doc_idx] = vector
                f['doc_ids'][doc_idx] = doc_id
        
        logger.info(f"Saved {len(document_vectors)} document vectors to {vectors_path}")
    
    def load_document_vectors(self, doc_ids: List[str] = None) -> Tuple[np.ndarray, List[str]]:
        """
        Load document vectors from disk.
        
        Args:
            doc_ids: Optional list of document IDs to load. If None, load all.
            
        Returns:
            Tuple of (document vectors, document IDs)
        """
        vectors_path = self.index_dir / "document_vectors.h5"
        
        if not vectors_path.exists():
            logger.error(f"Document vectors file not found: {vectors_path}")
            return np.array([]), []
        
        with h5py.File(vectors_path, 'r') as f:
            if doc_ids is None:
                # Load all vectors
                vectors = f['vectors'][:]
                ids = [str(id) for id in f['doc_ids'][:]]
            else:
                # Load specific vectors
                indices = [self.doc_id_to_idx[doc_id] for doc_id in doc_ids if doc_id in self.doc_id_to_idx]
                vectors = f['vectors'][indices]
                ids = [str(f['doc_ids'][i]) for i in indices]
        
        logger.info(f"Loaded {len(vectors)} document vectors")
        return vectors, ids
    
    def process_in_batches(self, document_generator, batch_size: int = 1000):
        """
        Process documents in batches to handle large datasets.
        
        Args:
            document_generator: Generator yielding batches of documents
            batch_size: Size of batches to process
        """
        # Initialize by fitting on first batch
        first_batch = next(document_generator)
        self.fit_vectorizers(first_batch)
        
        # Process first batch
        doc_vectors, term_vectors = self.fit_transform(first_batch)
        doc_ids = [doc['id'] for doc in first_batch]
        self.save_document_vectors(doc_vectors, doc_ids)
        
        # Process remaining batches
        for batch in document_generator:
            # Transform batch
            doc_vectors = self.transform(batch)
            doc_ids = [doc['id'] for doc in batch]
            
            # Save vectors
            self.save_document_vectors(doc_vectors, doc_ids)
        
        # Save the index
        self.save_index()

# Example usage:
if __name__ == "__main__":
    # This is a simple example to demonstrate how to use the IndexingEngine
    from document_processor import DocumentProcessor
    
    processor = DocumentProcessor(data_dir="./data")
    engine = IndexingEngine(index_dir="./index")
    
    # Process a sample batch
    sample_batch = next(processor.batch_document_generator())
    
    # Fit and transform
    doc_vectors, term_vectors = engine.fit_transform(sample_batch)
    
    print(f"Created document vectors with shape: {doc_vectors.shape}")
    print(f"Created term vectors with shape: {term_vectors.shape}")
    
    # Save the index
    engine.save_index()

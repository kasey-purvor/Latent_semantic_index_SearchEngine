import os
import logging
import numpy as np
import pickle
from typing import Dict, List, Any, Tuple, Set
from pathlib import Path
from keybert import KeyBERT
from tqdm import tqdm
import torch
import h5py

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("KeywordExtractor")

class KeywordExtractor:
    """
    Extracts keywords from academic papers using KeyBERT.
    
    This module handles keyword extraction during document indexing
    and provides methods to boost search relevance based on keyword matching.
    It's optimized for memory efficiency when processing large datasets.
    """
    
    def __init__(self, 
                 index_dir: str,
                 model_name: str = "all-MiniLM-L6-v2",
                 top_n: int = 5,
                 keyphrase_ngram_range: Tuple[int, int] = (1, 3),
                 batch_size: int = 32):
        """
        Initialize the keyword extractor.
        
        Args:
            index_dir: Directory to store keyword data
            model_name: Name of the sentence transformer model to use
            top_n: Number of keywords to extract per document
            keyphrase_ngram_range: Range of ngrams to consider for keyphrases
            batch_size: Number of documents to process at once
        """
        self.index_dir = Path(index_dir)
        self.model_name = model_name
        self.top_n = top_n
        self.keyphrase_ngram_range = keyphrase_ngram_range
        self.batch_size = batch_size
        
        # Initialize KeyBERT model
        logger.info(f"Initializing KeyBERT with model: {model_name}")
        self.model = KeyBERT(model=model_name)
        
        # Document keyword mapping: doc_id -> list of (keyword, score) tuples
        self.doc_keywords = {}
        
        # Ensure index directory exists
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"KeywordExtractor initialized with top_n={top_n}, ngram_range={keyphrase_ngram_range}")
    
    def extract_keywords(self, documents: List[Dict[str, Any]]) -> Dict[str, List[Tuple[str, float]]]:
        """
        Extract keywords from a batch of documents.
        
        Args:
            documents: List of processed documents
            
        Returns:
            Dictionary mapping document IDs to keyword lists
        """
        results = {}
        
        # Process documents in smaller batches to manage memory
        for i in range(0, len(documents), self.batch_size):
            batch = documents[i:i+self.batch_size]
            
            # Extract abstracts as they typically contain the most relevant information
            abstracts = [doc.get('abstract', doc.get('title', '')) for doc in batch]
            doc_ids = [doc['id'] for doc in batch]
            
            logger.info(f"Extracting keywords from batch of {len(batch)} documents")
            
            # Use KeyBERT to extract keywords
            batch_keywords = self.model.extract_keywords(
                abstracts,
                keyphrase_ngram_range=self.keyphrase_ngram_range,
                stop_words='english',
                use_maxsum=True,  # Use maxsum to get more diverse keywords
                top_n=self.top_n
            )
            
            # Store results
            for doc_id, keywords in zip(doc_ids, batch_keywords):
                results[doc_id] = keywords
                self.doc_keywords[doc_id] = keywords
            
            # Clear GPU cache periodically
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        logger.info(f"Extracted keywords for {len(results)} documents")
        return results
    
    def save_keywords(self):
        """Save extracted keywords to disk."""
        keyword_path = self.index_dir / "keywords.pkl"
        
        with open(keyword_path, 'wb') as f:
            pickle.dump(self.doc_keywords, f)
        
        logger.info(f"Saved keywords for {len(self.doc_keywords)} documents to {keyword_path}")
    
    def load_keywords(self):
        """Load extracted keywords from disk."""
        keyword_path = self.index_dir / "keywords.pkl"
        
        if not keyword_path.exists():
            logger.error(f"Keywords file not found: {keyword_path}")
            return
        
        with open(keyword_path, 'rb') as f:
            self.doc_keywords = pickle.load(f)
        
        logger.info(f"Loaded keywords for {len(self.doc_keywords)} documents from {keyword_path}")
    
    def get_document_keywords(self, doc_id: str) -> List[Tuple[str, float]]:
        """
        Get keywords for a specific document.
        
        Args:
            doc_id: Document ID
            
        Returns:
            List of (keyword, score) tuples
        """
        return self.doc_keywords.get(doc_id, [])
    
    def calculate_keyword_boost(self, query: str, doc_id: str) -> float:
        """
        Calculate a relevance boost based on keyword matching.
        
        Args:
            query: Search query
            doc_id: Document ID
            
        Returns:
            Boost factor for document relevance
        """
        if doc_id not in self.doc_keywords:
            return 1.0
        
        # Normalize query
        query = query.lower()
        query_terms = set(query.split())
        
        # Get document keywords
        doc_keywords = self.doc_keywords[doc_id]
        
        # Calculate score based on matching keywords
        max_boost = 2.0
        match_score = 0.0
        
        for keyword, score in doc_keywords:
            # Check if keyword appears in query
            keyword_terms = set(keyword.lower().split())
            if keyword_terms.intersection(query_terms):
                match_score += score
        
        # Scale the boost factor, capped at max_boost
        boost = 1.0 + min(match_score, max_boost - 1.0)
        
        return boost
    
    def expand_query(self, query: str, top_docs: List[str], max_terms: int = 3) -> str:
        """
        Expand a query with keywords from top-ranked documents.
        
        Args:
            query: Original query
            top_docs: List of top document IDs from initial search
            max_terms: Maximum number of terms to add
            
        Returns:
            Expanded query string
        """
        # Collect keywords from top documents
        expansion_terms = {}
        
        for doc_id in top_docs:
            if doc_id in self.doc_keywords:
                for keyword, score in self.doc_keywords[doc_id]:
                    if keyword.lower() not in query.lower():
                        if keyword in expansion_terms:
                            expansion_terms[keyword] += score
                        else:
                            expansion_terms[keyword] = score
        
        # Sort by score and take top terms
        sorted_terms = sorted(expansion_terms.items(), key=lambda x: x[1], reverse=True)[:max_terms]
        
        # Add to original query
        expanded_query = query
        for term, _ in sorted_terms:
            expanded_query += f" {term}"
        
        logger.info(f"Expanded query: '{query}' -> '{expanded_query}'")
        return expanded_query
    
    def process_in_batches(self, document_generator):
        """
        Process documents in batches to handle large datasets.
        
        Args:
            document_generator: Generator yielding batches of documents
        """
        for batch in document_generator:
            # Extract keywords
            self.extract_keywords(batch)
            
            # Periodically save to disk
            if len(self.doc_keywords) % 10000 == 0:
                self.save_keywords()
        
        # Final save
        self.save_keywords()

# Example usage:
if __name__ == "__main__":
    # This is a simple example to demonstrate how to use the KeywordExtractor
    from document_processor import DocumentProcessor
    
    processor = DocumentProcessor(data_dir="./data")
    extractor = KeywordExtractor(index_dir="./index")
    
    # Process a sample batch
    sample_batch = next(processor.batch_document_generator())
    
    # Extract keywords
    keywords = extractor.extract_keywords(sample_batch)
    
    # Print some examples
    for i, doc in enumerate(sample_batch[:3]):
        print(f"Document: {doc['title']}")
        print(f"Keywords: {keywords[doc['id']]}")
        print()
    
    # Save keywords
    extractor.save_keywords()

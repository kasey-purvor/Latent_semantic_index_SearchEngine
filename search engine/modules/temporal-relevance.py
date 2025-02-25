import logging
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path
import pickle
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TemporalRelevance")

class TemporalRelevanceAdjuster:
    """
    Adjusts document relevance based on publication year.
    
    This module provides methods to boost or penalize search results
    based on recency, allowing users to control the importance of
    document age in search results.
    """
    
    def __init__(self, 
                 index_dir: str,
                 default_recency_preference: float = 0.3,
                 max_age_factor: int = 10):
        """
        Initialize the temporal relevance adjuster.
        
        Args:
            index_dir: Directory to store temporal data
            default_recency_preference: Default weight for recency (0.0 to 1.0)
                0.0 = no preference for recent documents
                1.0 = strong preference for recent documents
            max_age_factor: Maximum factor for age normalization
        """
        self.index_dir = Path(index_dir)
        self.default_recency_preference = default_recency_preference
        self.max_age_factor = max_age_factor
        
        # Document year mapping: doc_id -> publication year
        self.doc_years = {}
        
        # Current year (for age calculation)
        self.current_year = datetime.now().year
        
        # Ensure index directory exists
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"TemporalRelevanceAdjuster initialized with recency_preference={default_recency_preference}")
    
    def index_document_years(self, documents: List[Dict[str, Any]]):
        """
        Index publication years for a batch of documents.
        
        Args:
            documents: List of processed documents
        """
        for doc in documents:
            year = doc.get('year', 0)
            if year > 0:  # Skip documents with invalid years
                self.doc_years[doc['id']] = year
        
        logger.info(f"Indexed years for {len(documents)} documents")
    
    def save_years_index(self):
        """Save document years index to disk."""
        years_path = self.index_dir / "doc_years.pkl"
        
        with open(years_path, 'wb') as f:
            pickle.dump(self.doc_years, f)
        
        logger.info(f"Saved years for {len(self.doc_years)} documents to {years_path}")
    
    def load_years_index(self):
        """Load document years index from disk."""
        years_path = self.index_dir / "doc_years.pkl"
        
        if not years_path.exists():
            logger.error(f"Document years file not found: {years_path}")
            return
        
        with open(years_path, 'rb') as f:
            self.doc_years = pickle.load(f)
        
        logger.info(f"Loaded years for {len(self.doc_years)} documents from {years_path}")
    
    def calculate_temporal_boost(self, 
                                doc_id: str, 
                                recency_preference: Optional[float] = None) -> float:
        """
        Calculate temporal boost factor for a document.
        
        Args:
            doc_id: Document ID
            recency_preference: Override default recency preference (0.0 to 1.0)
            
        Returns:
            Boost factor for document relevance
        """
        if doc_id not in self.doc_years:
            return 1.0
        
        # Use provided recency preference or default
        recency_pref = recency_preference if recency_preference is not None else self.default_recency_preference
        
        # Cap recency preference between 0 and 1
        recency_pref = max(0.0, min(1.0, recency_pref))
        
        # Calculate document age
        doc_year = self.doc_years[doc_id]
        age = self.current_year - doc_year
        
        # Normalize age by max_age_factor (typically 10 years)
        normalized_age = min(age, self.max_age_factor) / self.max_age_factor
        
        # Calculate boost factor:
        # - recency_pref = 0: boost = 1.0 (no effect)
        # - recency_pref = 1, new doc: boost = 1.0
        # - recency_pref = 1, old doc: boost = 0.5
        boost = 1.0 / (1.0 + recency_pref * normalized_age)
        
        return boost
    
    def apply_temporal_boost(self, 
                            search_results: List[Dict[str, Any]], 
                            recency_preference: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Apply temporal boost to search results.
        
        Args:
            search_results: List of search result dictionaries with 'id' and 'score' keys
            recency_preference: Override default recency preference
            
        Returns:
            List of search results with adjusted scores
        """
        for result in search_results:
            doc_id = result['id']
            original_score = result['score']
            
            # Calculate temporal boost
            boost = self.calculate_temporal_boost(doc_id, recency_preference)
            
            # Apply boost to score
            adjusted_score = original_score * boost
            
            # Update result
            result['score'] = adjusted_score
            result['temporal_boost'] = boost
        
        # Re-sort results by adjusted score
        search_results.sort(key=lambda x: x['score'], reverse=True)
        
        return search_results
    
    def get_year_distribution(self) -> Dict[int, int]:
        """
        Get distribution of documents by year.
        
        Returns:
            Dictionary mapping years to document counts
        """
        distribution = {}
        
        for year in self.doc_years.values():
            if year in distribution:
                distribution[year] += 1
            else:
                distribution[year] = 1
        
        return distribution
    
    def get_document_age(self, doc_id: str) -> int:
        """
        Get age of a document in years.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Age in years or -1 if unknown
        """
        if doc_id not in self.doc_years:
            return -1
        
        return self.current_year - self.doc_years[doc_id]
    
    def process_in_batches(self, document_generator):
        """
        Process documents in batches to handle large datasets.
        
        Args:
            document_generator: Generator yielding batches of documents
        """
        for batch in document_generator:
            # Index document years
            self.index_document_years(batch)
            
            # Periodically save to disk
            if len(self.doc_years) % 10000 == 0:
                self.save_years_index()
        
        # Final save
        self.save_years_index()

# Example usage:
if __name__ == "__main__":
    # This is a simple example to demonstrate how to use the TemporalRelevanceAdjuster
    from document_processor import DocumentProcessor
    
    processor = DocumentProcessor(data_dir="./data")
    adjuster = TemporalRelevanceAdjuster(index_dir="./index")
    
    # Process a sample batch
    sample_batch = next(processor.batch_document_generator())
    
    # Index document years
    adjuster.index_document_years(sample_batch)
    
    # Example of adjusting search results
    mock_results = [
        {'id': sample_batch[0]['id'], 'score': 0.9},
        {'id': sample_batch[1]['id'], 'score': 0.8},
        {'id': sample_batch[2]['id'], 'score': 0.7}
    ]
    
    # Print years
    for result in mock_results:
        doc_id = result['id']
        year = adjuster.doc_years.get(doc_id, "Unknown")
        print(f"Document {doc_id}: Year {year}")
    
    # Apply temporal boost with different preferences
    print("\nNo recency preference (0.0):")
    adjusted_results = adjuster.apply_temporal_boost(mock_results.copy(), recency_preference=0.0)
    for result in adjusted_results:
        print(f"Document {result['id']}: Score {result['score']:.4f}, Boost {result['temporal_boost']:.4f}")
    
    print("\nStrong recency preference (1.0):")
    adjusted_results = adjuster.apply_temporal_boost(mock_results.copy(), recency_preference=1.0)
    for result in adjusted_results:
        print(f"Document {result['id']}: Score {result['score']:.4f}, Boost {result['temporal_boost']:.4f}")
    
    # Save index
    adjuster.save_years_index()

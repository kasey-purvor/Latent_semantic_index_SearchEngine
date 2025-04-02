"""
Base classes for indexing implementations.
Provides common functionality and interfaces for all indexers.
"""

import os
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from tqdm import tqdm
import numpy as np
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class BaseIndexer(ABC):
    """Base class for all index implementations."""
    
    def __init__(self, index_name: str):
        """
        Initialize the base indexer.
        
        Args:
            index_name: Name of the index type (e.g., 'bm25', 'lsi_basic')
        """
        self.index_name = index_name
        self.DEFAULT_FIELD_WEIGHTS = {
            'title': 2.5,
            'abstract': 1.5,
            'body': 1.0,
            'topics': 2.0,
            'keywords': 3.0
        }
        self._index = None
        self._documents = []
        self._metadata = {}
        
    def _print_progress(self, current: int, total: int, prefix: str = "") -> None:
        """
        Print progress that overwrites the current line.
        
        Args:
            current: Current progress count
            total: Total items to process
            prefix: Optional prefix for the progress message
        """
        percentage = (current / total) * 100 if total > 0 else 0
        print(f"\r{prefix}Progress: {current}/{total} ({percentage:.1f}%)", end='', flush=True)
        
    def _save_metadata(self, output_dir: str) -> None:
        """
        Save index metadata.
        
        Args:
            output_dir: Directory to save metadata
        """
        metadata_path = os.path.join(output_dir, f"{self.index_name}_metadata.joblib")
        joblib.dump(self._metadata, metadata_path)
        
    def _load_metadata(self, index_dir: str) -> None:
        """
        Load index metadata.
        
        Args:
            index_dir: Directory containing metadata
        """
        metadata_path = os.path.join(index_dir, f"{self.index_name}_metadata.joblib")
        if os.path.exists(metadata_path):
            self._metadata = joblib.load(metadata_path)
            
    @abstractmethod
    def build_index(self, documents: List[Dict[str, Any]], output_dir: str) -> None:
        """
        Build the index from documents.
        
        Args:
            documents: List of document dictionaries
            output_dir: Directory to save the index
        """
        pass
        
    @abstractmethod
    def load_index(self, index_dir: str) -> None:
        """
        Load an existing index.
        
        Args:
            index_dir: Directory containing the index
        """
        pass
        
    @abstractmethod
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search using the index.
        
        Args:
            query: Search query string
            top_k: Number of results to return
            
        Returns:
            List of search results with scores
        """
        pass
        
    def get_index_info(self) -> Dict[str, Any]:
        """
        Get information about the current index.
        
        Returns:
            Dictionary containing index information
        """
        return {
            'index_name': self.index_name,
            'num_documents': len(self._documents),
            'metadata': self._metadata
        } 
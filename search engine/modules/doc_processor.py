import json
import os
import re
import logging
from typing import Dict, List, Any, Iterator, Optional, Tuple
from pathlib import Path
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DocumentProcessor")

class DocumentProcessor:
    """
    Processes academic paper documents from JSON format.
    
    This class handles the extraction, normalization, and batching of documents
    for efficient processing. It's designed to work with large datasets by
    streaming documents rather than loading everything into memory.
    """
    
    def __init__(self, 
                 data_dir: str, 
                 batch_size: int = 1000,
                 min_token_length: int = 3,
                 max_token_length: int = 30):
        """
        Initialize the document processor.
        
        Args:
            data_dir: Directory containing JSON document files
            batch_size: Number of documents to process in a batch
            min_token_length: Minimum length of tokens to keep
            max_token_length: Maximum length of tokens to keep
        """
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.min_token_length = min_token_length
        self.max_token_length = max_token_length
        
        # Compile regex patterns for text normalization
        self.word_pattern = re.compile(r'\b[a-zA-Z][a-zA-Z0-9]*\b')
        self.whitespace_pattern = re.compile(r'\s+')
        
        # Verify the data directory exists
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory {data_dir} not found")
            
        logger.info(f"DocumentProcessor initialized with data directory: {data_dir}")
        
    def list_document_files(self) -> List[Path]:
        """Get a list of all JSON document files in the data directory."""
        json_files = list(self.data_dir.glob("**/*.json"))
        logger.info(f"Found {len(json_files)} JSON files in {self.data_dir}")
        return json_files
        
    def normalize_text(self, text: str) -> str:
        """
        Normalize text by converting to lowercase, removing special characters,
        and standardizing whitespace.
        """
        if not text:
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Extract only valid words
        words = self.word_pattern.findall(text)
        
        # Filter by length
        words = [word for word in words if self.min_token_length <= len(word) <= self.max_token_length]
        
        # Join with single spaces
        return " ".join(words)
    
    def extract_document_fields(self, doc: Dict[str, Any]) -> Dict[str, str]:
        """
        Extract and normalize relevant fields from a document.
        
        Args:
            doc: A dictionary containing document data
            
        Returns:
            Dictionary with normalized text fields
        """
        # Extract fields with proper error handling
        title = doc.get('title', '')
        abstract = doc.get('abstract', '')
        
        # For full text, we might need to handle different key names or formats
        full_text = doc.get('fullText', doc.get('full_text', ''))
        
        # Normalize each field
        normalized = {
            'id': doc.get('identifiers', [''])[0] if doc.get('identifiers') else str(hash(title)),
            'title': self.normalize_text(title),
            'abstract': self.normalize_text(abstract),
            'full_text': self.normalize_text(full_text),
            'year': doc.get('year', 0),
            'authors': doc.get('authors', []),
            'journal': doc.get('journals', [{}])[0].get('title', '') if doc.get('journals') else '',
            'language': doc.get('language', {}).get('code', 'en') if isinstance(doc.get('language'), dict) else 'en'
        }
        
        return normalized
    
    def process_document(self, file_path: Path) -> Dict[str, Any]:
        """
        Process a single document file.
        
        Args:
            file_path: Path to the JSON document file
            
        Returns:
            Processed document or None if an error occurred
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                doc = json.load(f)
            return self.extract_document_fields(doc)
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            return None
    
    def process_documents_batch(self, files: List[Path]) -> List[Dict[str, Any]]:
        """
        Process a batch of document files.
        
        Args:
            files: List of paths to JSON document files
            
        Returns:
            List of processed documents
        """
        processed_docs = []
        for file_path in tqdm(files, desc="Processing batch"):
            doc = self.process_document(file_path)
            if doc:
                processed_docs.append(doc)
        
        return processed_docs
    
    def document_generator(self) -> Iterator[Dict[str, Any]]:
        """
        Generator that yields processed documents one at a time.
        
        This is memory-efficient for processing large datasets.
        """
        files = self.list_document_files()
        
        for file_path in files:
            doc = self.process_document(file_path)
            if doc:
                yield doc
    
    def batch_document_generator(self) -> Iterator[List[Dict[str, Any]]]:
        """
        Generator that yields batches of processed documents.
        
        This is memory-efficient while also allowing batch processing.
        """
        files = self.list_document_files()
        
        for i in range(0, len(files), self.batch_size):
            batch_files = files[i:i+self.batch_size]
            batch_docs = self.process_documents_batch(batch_files)
            
            if batch_docs:
                logger.info(f"Yielding batch of {len(batch_docs)} documents")
                yield batch_docs
    
    def get_document_statistics(self, sample_size: int = 1000) -> Dict[str, Any]:
        """
        Get statistics about the document collection to inform processing decisions.
        
        Args:
            sample_size: Number of documents to sample for statistics
            
        Returns:
            Dictionary with statistics
        """
        # Sample documents
        files = self.list_document_files()
        sample_files = files[:min(sample_size, len(files))]
        
        sample_docs = self.process_documents_batch(sample_files)
        
        # Calculate statistics
        stats = {
            'total_documents': len(files),
            'sample_size': len(sample_docs),
            'avg_title_length': np.mean([len(doc['title'].split()) for doc in sample_docs]),
            'avg_abstract_length': np.mean([len(doc['abstract'].split()) for doc in sample_docs]),
            'avg_fulltext_length': np.mean([len(doc['full_text'].split()) for doc in sample_docs]),
            'year_distribution': {}
        }
        
        # Count documents by year
        for doc in sample_docs:
            year = doc['year']
            if year in stats['year_distribution']:
                stats['year_distribution'][year] += 1
            else:
                stats['year_distribution'][year] = 1
                
        return stats

# Example usage:
if __name__ == "__main__":
    # This is a simple example to demonstrate how to use the DocumentProcessor
    processor = DocumentProcessor(data_dir="./data")
    
    # Get statistics
    stats = processor.get_document_statistics(sample_size=100)
    print("Document Statistics:")
    print(f"Total documents: {stats['total_documents']}")
    print(f"Average title length: {stats['avg_title_length']:.2f} words")
    print(f"Average abstract length: {stats['avg_abstract_length']:.2f} words")
    
    # Process one batch
    for batch in processor.batch_document_generator():
        print(f"Processed batch of {len(batch)} documents")
        # Just process one batch for this example
        break
"""
BERT-enhanced LSI implementation.
Extends field-weighted LSI with BERT keyword extraction for better semantic understanding.
"""

import os
import numpy as np
import joblib
import logging
import torch
from typing import Dict, List, Any, Tuple, Optional
from tqdm import tqdm

from keybert import KeyBERT
from sentence_transformers import SentenceTransformer

from .field_weighted_lsi import FieldWeightedLSIIndexer

class BertEnhancedIndexer(FieldWeightedLSIIndexer):
    """BERT-enhanced LSI indexer implementation that adds keyword extraction to field-weighted LSI."""
    
    def __init__(self, n_components: int = 150):
        """
        Initialize the BERT-enhanced LSI indexer.
        
        Args:
            n_components: Number of latent semantic dimensions
        """
        # Initialize with the parent class
        super().__init__(n_components)
        self.index_name = "lsi_bert_enhanced"
        self.bert_model = None
        self.keybert_model = None
        
        # Add 'keywords' to available fields
        if 'keywords' not in self._available_field_names:
            self._available_field_names.append('keywords')
    
    def build_index(self, documents: Dict[str, List], output_dir: str) -> None:
        """
        Build a BERT-enhanced LSI index from documents.
        
        Args:
            documents: Dictionary containing document fields
            output_dir: Directory to save the index
        """
        logging.info(f"Building BERT-enhanced LSI index with {self.n_components} dimensions")
        
        # Check if CUDA is available
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logging.info(f"Using device: {device} for BERT models")
        
        # Initialize BERT model
        logging.info("Initializing BERT models")
        try:
            self.bert_model = SentenceTransformer('paraphrase-mpnet-base-v2')
            self.bert_model = self.bert_model.to(device)
            self.keybert_model = KeyBERT(model=self.bert_model)
        except Exception as e:
            logging.error(f"Error initializing BERT models: {e}")
            raise
        
        # Extract keywords using KeyBERT and add them to documents
        logging.info("Extracting keywords with KeyBERT")
        documents['keywords'] = self._extract_keywords(documents)
        
        # Call the parent class to build the field-weighted index
        # This will create a single vocabulary on all fields (including keywords)
        # and then apply adaptive weighting
        super().build_index(documents, output_dir)
        
        # Update metadata to reflect BERT enhancement
        self._metadata['index_type'] = self.index_name
        self._metadata['bert_model'] = 'paraphrase-mpnet-base-v2'
        
        # Save updated metadata
        self._save_metadata(output_dir)
    
    def _extract_keywords(self, documents: Dict[str, List]) -> List[str]:
        """
        Extract keywords from documents using KeyBERT.
        
        Args:
            documents: Dictionary containing document fields
            
        Returns:
            List of keyword strings for each document
        """
        if self.keybert_model is None:
            raise ValueError("BERT model must be initialized before extracting keywords")
        
        # Combine title and abstract for keyword extraction
        combined_texts = []
        for i in range(len(documents['paper_ids'])):
            title = documents['titles'][i] if i < len(documents['titles']) and documents['titles'][i] else ""
            abstract = documents['abstracts'][i] if i < len(documents['abstracts']) and documents['abstracts'][i] else ""
            combined_texts.append(f"{title} {abstract}".strip())
        
        # Process documents in batches
        batch_size = 32  # Adjust based on available VRAM
        keywords_list = []
        
        for i in tqdm(range(0, len(combined_texts), batch_size), desc="Extracting keywords"):
            batch = combined_texts[i:i+batch_size]
            
            # Skip empty batches
            if not batch:
                keywords_list.extend(["" for _ in range(min(batch_size, len(combined_texts) - i))])
                continue
            
            try:
                # Extract keywords for this batch
                batch_keywords = self.keybert_model.extract_keywords(
                    docs=batch,
                    keyphrase_ngram_range=(1, 1),
                    stop_words='english',
                    use_mmr=True,
                    diversity=0.7,
                    top_n=10
                )
                
                # Process each document's keywords
                for doc_keywords in batch_keywords:
                    doc_filtered_keywords = []
                    
                    if isinstance(doc_keywords, list):
                        for item in doc_keywords:
                            if isinstance(item, tuple) and len(item) == 2:
                                keyword, score = item
                                if score > 0.35:  # Filter by score threshold
                                    doc_filtered_keywords.append(str(keyword))
                    
                    # Join keywords into a space-separated string
                    keywords_list.append(" ".join(doc_filtered_keywords))
                
            except Exception as e:
                logging.error(f"Error extracting keywords for batch: {e}")
                keywords_list.extend(["" for _ in range(len(batch))])
        
        # Ensure we have the right number of keyword lists
        while len(keywords_list) < len(documents['paper_ids']):
            keywords_list.append("")
        
        # Trim if we have too many
        if len(keywords_list) > len(documents['paper_ids']):
            keywords_list = keywords_list[:len(documents['paper_ids'])]
        
        logging.info(f"Extracted keywords for {len(keywords_list)} documents")
        return keywords_list 
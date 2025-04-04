"""
Utility functions for document processing and indexing.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from tqdm import tqdm

def load_papers(directory_path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Load papers from JSON files in a directory structure.
    
    Args:
        directory_path: Path to the directory containing JSON files
        limit: Maximum number of papers to load (for testing)
        
    Returns:
        List of paper dictionaries
    """
    papers = []
    logging.info(f"Loading papers from {directory_path}")
    
    counter = 0
    for root, _, files in os.walk(directory_path):
        for file in files:
            if limit and counter >= limit:
                break
                
            if file.endswith('.json'):
                counter += 1
                print(f"\rProcessing: {counter}", end='', flush=True)
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        paper_data = json.load(f)
                    
                    # Add file path information for reference
                    paper_data['file_path'] = file_path
                    papers.append(paper_data)
                                        
                except Exception as e:  
                    logging.error(f"Error loading {file_path}: {e}")
        
        if limit and counter >= limit:
            break
            
    logging.info(f"Finished loading {counter} papers")
    return papers

def extract_fields(papers: List[Dict[str, Any]]) -> Dict[str, List]:
    """
    Extract title, abstract, body text, and topics from papers.
    
    Args:
        papers: List of paper dictionaries
        
    Returns:
        Dictionary of extracted fields and paper IDs
    """
    titles = []
    abstracts = []
    bodies = []
    paper_ids = []
    topics_list = []
    counter = 0
    
    for paper in papers:
        counter += 1
        print(f"\rProcessing: {counter}", end='', flush=True)
        
        # Extract ID
        paper_id = paper.get('coreId')
        paper_ids.append(paper_id)  
        
        # Extract title - handle missing cases
        title = paper.get('title')
        titles.append(title if isinstance(title, str) and title.strip() != '' else '')
        
        # Extract abstract - handle None values
        abstract = paper.get('abstract')
        abstracts.append(abstract if isinstance(abstract, str) and abstract.strip() != '' else '')
        
        # Extract fullText - handle None values and empty strings
        fulltext = paper.get('fullText')
        bodies.append(fulltext if isinstance(fulltext, str) and fulltext.strip() != '' else '')
        
        # Extract topics - handle empty lists
        topic_data = paper.get('topics')
        if isinstance(topic_data, list) and len(topic_data) > 0:
            # Convert topics list to string to ensure consistent handling
            topics_list.append(', '.join(str(topic) for topic in topic_data))
        else:
            topics_list.append('')
    
    return {
        'paper_ids': paper_ids,
        'titles': titles,
        'abstracts': abstracts,
        'bodies': bodies,
        'topics': topics_list
    }

def combine_fields(
    fields: Dict[str, List], 
    field_weights: Optional[Dict[str, float]] = None,
    normalize: bool = True
) -> List[str]:
    """
    Combine document fields.
    
    When field_weights is provided, applies weights to each field.
    Otherwise, simply concatenates the fields with a space separator.
    
    Args:
        fields: Dictionary of field lists
        field_weights: Optional dictionary of field weights
        normalize: Whether to normalize the weights (only used when weights are provided)
        
    Returns:
        List of combined field texts
    """
    num_docs = len(fields['paper_ids'])
    
    # Simple concatenation mode (for basic LSI)
    if field_weights is None:
        combined_texts = []
        for i in range(num_docs):
            parts = []
            # Add each field if it exists and is not empty
            for field_name in ['titles', 'abstracts', 'bodies', 'topics', 'keywords']:
                if field_name in fields and i < len(fields[field_name]) and fields[field_name][i]:
                    parts.append(fields[field_name][i])
            combined_texts.append(' '.join(parts))
        return combined_texts
    
    # Field weighting mode (for field-weighted LSI)
    # Normalize weights if requested
    if normalize:
        total_weight = sum(field_weights.values())
        field_weights = {k: v/total_weight for k, v in field_weights.items()}
    
    combined_texts = []
    
    for i in range(num_docs):
        combined_text = []
        
        # Add each field with its weight
        for field_name, weight in field_weights.items():
            # Handle singular/plural field name conversion
            field_list_name = f"{field_name}s" if not field_name.endswith('s') else field_name
            
            if field_list_name in fields and i < len(fields[field_list_name]) and fields[field_list_name][i]:
                # Repeat the field text based on its weight
                field_text = fields[field_list_name][i]
                combined_text.extend([field_text] * int(weight * 10))  # Scale weight for better granularity
        
        combined_texts.append(' '.join(combined_text))
    
    return combined_texts

def create_output_dirs(base_dir: str, index_name: str) -> str:
    """
    Create output directories for an index.
    
    Args:
        base_dir: Base directory for all indices
        index_name: Name of the specific index
        
    Returns:
        Path to the created index directory
    """
    index_dir = os.path.join(base_dir, index_name)
    os.makedirs(index_dir, exist_ok=True)
    return index_dir 
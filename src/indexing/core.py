"""
Core functionality for the indexing module.
Provides functions for loading JSON papers and creating LSI vectors.
"""

import os
import json
import numpy as np
import logging
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional, Union
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from scipy.sparse import spmatrix
import joblib

# KeyBERT imports
import torch
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Default field weights
DEFAULT_FIELD_WEIGHTS = {
    'title': 2.5,
    'abstract': 1.5,
    'body': 1.0,
    'topics': 2.0,
    'keywords': 3.0  # Weight for extracted keywords from KeyBERT
}

def load_papers(directory_path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Load papers from JSON files in a directory structure
    
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
    Extract title, abstract, body text, and topics from papers
    
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

def extract_keywords_with_keybert(extracted_data: Dict[str, List]) -> List[str]:
    """
    Extract keywords from documents using KeyBERT
    
    Args:
        extracted_data: Dictionary containing extracted fields
        
    Returns:
        List of keyword strings for each document
    """
    # Initialize KeyBERT with a sentence transformer model
    logging.info("Initializing KeyBERT for keyword extraction")
    
    # Check if CUDA is available
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device} for KeyBERT")
    
    # Initialize the model - KeyBERT accepts SentenceTransformer models directly
    try:
        # Properly handle model initialization
        sentence_model = SentenceTransformer('paraphrase-mpnet-base-v2')
        sentence_model = sentence_model.to(device)
        # KeyBERT expects a model instance but there may be type issues
        kw_model = KeyBERT(model=sentence_model)
    except Exception as e:
        logging.error(f"Error initializing KeyBERT model: {e}")
        raise
    
    # Combine text fields for each document
    combined_texts = []
    for i in range(len(extracted_data['paper_ids'])):
        # Combine title, abstract, and body with spacing between them
        doc_text = (
            extracted_data['titles'][i] + " " + 
            extracted_data['abstracts'][i] + " " + 
            extracted_data['bodies'][i]
        ).strip()
        combined_texts.append(doc_text)
    
    # Process in batches to maximize GPU usage
    batch_size = 100  # Adjust based on available VRAM
    all_keywords_lists = []
    
    # Show progress bar
    logging.info(f"Extracting keywords from {len(combined_texts)} documents")
    
    for i in tqdm(range(0, len(combined_texts), batch_size), desc="Extracting keywords"):
        batch = combined_texts[i:i+batch_size]
        # Filter out empty documents
        batch = [doc for doc in batch if doc and len(doc) > 50]  # Minimum text length
        
        if not batch:  # Skip empty batches
            # Add empty strings for skipped documents
            all_keywords_lists.extend(["" for _ in range(min(batch_size, len(combined_texts) - i))])
            continue
        
        try:
            # Use correct parameter name 'docs' instead of 'documents'
            batch_keywords = kw_model.extract_keywords(
                docs=batch,
                keyphrase_ngram_range=(1, 3),  # Extract 1-3 word keyphrases
                stop_words='english',
                use_mmr=True,         # Use Maximal Marginal Relevance
                diversity=0.7,        # Higher diversity
                top_n=10              # Extract top 10 keywords
            )
            
            # Process each document's keywords
            for doc_keywords in batch_keywords:
                doc_filtered_keywords = []
                
                # Handle different possible structures
                if isinstance(doc_keywords, list):
                    for item in doc_keywords:
                        if isinstance(item, tuple) and len(item) == 2:
                            keyword, score = item
                            if isinstance(score, (int, float)) and score > 0.35:
                                doc_filtered_keywords.append(str(keyword))
                elif isinstance(doc_keywords, tuple) and len(doc_keywords) == 2:
                    keyword, score = doc_keywords
                    if isinstance(score, (int, float)) and score > 0.35:
                        doc_filtered_keywords.append(str(keyword))
                
                # Join keywords into a space-separated string
                all_keywords_lists.append(" ".join(doc_filtered_keywords))
                
        except Exception as e:
            logging.error(f"Error extracting keywords for batch: {e}")
            # Add empty strings on error
            all_keywords_lists.extend(["" for _ in range(len(batch))])
    
    # Ensure we have the right number of keyword lists
    if len(all_keywords_lists) < len(extracted_data['paper_ids']):
        # Pad with empty strings if necessary
        all_keywords_lists.extend(["" for _ in range(len(extracted_data['paper_ids']) - len(all_keywords_lists))])
    elif len(all_keywords_lists) > len(extracted_data['paper_ids']):
        # Truncate if we have too many
        all_keywords_lists = all_keywords_lists[:len(extracted_data['paper_ids'])]
    
    logging.info(f"Extracted keywords for {len(all_keywords_lists)} documents")
    return all_keywords_lists

def create_tfidf_vectors(extracted_data: Dict[str, List], keyword_texts: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Create TF-IDF vectors for each field
    
    Args:
        extracted_data: Dictionary containing extracted fields
        keyword_texts: Optional list of keyword texts from KeyBERT
        
    Returns:
        Dictionary with vectorizer and field vectors
    """
    # Create and configure the vectorizer
    vectorizer = TfidfVectorizer(
        min_df=2,                  # Ignore terms that appear in fewer than 2 documents
        max_df=0.95,               # Ignore terms that appear in more than 95% of documents
        stop_words='english',      # Remove common English stop words
        lowercase=True,            # Convert all text to lowercase
        strip_accents='unicode',   # Remove accents
        token_pattern=r'\b[a-zA-Z]{3,}\b'  # Only include words with 3+ letters
    )
    
    # Process topics to ensure they're strings, not lists
    processed_topics = []
    for topic in extracted_data['topics']:
        processed_topics.append(topic if isinstance(topic, str) else '')
    
    # Prepare all text for fitting the vectorizer
    all_text = (
        extracted_data['titles'] + 
        extracted_data['abstracts'] + 
        extracted_data['bodies'] + 
        processed_topics
    )
    
    # Add keyword texts if available
    if keyword_texts:
        all_text = all_text + keyword_texts
        
    vectorizer.fit(all_text)
    
    # Transform each field separately
    title_vectors = vectorizer.transform(extracted_data['titles'])
    logging.info(f'Title vectors processed: {title_vectors.shape} (documents × features)')
    
    abstract_vectors = vectorizer.transform(extracted_data['abstracts'])
    logging.info(f'Abstract vectors processed: {abstract_vectors.shape} (documents × features)')
    
    body_vectors = vectorizer.transform(extracted_data['bodies'])
    logging.info(f'Body vectors processed: {body_vectors.shape} (documents × features)')
    
    topic_vectors = vectorizer.transform(processed_topics)
    logging.info(f'Topic vectors processed: {topic_vectors.shape} (documents × features)')
    
    # Process keyword vectors if available
    keyword_vectors = None
    if keyword_texts:
        keyword_vectors = vectorizer.transform(keyword_texts)
        logging.info(f'Keyword vectors processed: {keyword_vectors.shape} (documents × features)')
    
    # Log the number of features (vocabulary size)
    feature_count = len(vectorizer.get_feature_names_out())
    logging.info(f"Created TF-IDF vectors with {feature_count} features")
    
    result = {
        'vectorizer': vectorizer,
        'title_vectors': title_vectors,
        'abstract_vectors': abstract_vectors,
        'body_vectors': body_vectors,
        'topic_vectors': topic_vectors
    }
    
    if keyword_vectors is not None:
        result['keyword_vectors'] = keyword_vectors
        
    return result

def apply_field_weighting(
    tfidf_data: Dict[str, Any], 
    field_weights: Dict[str, float] = DEFAULT_FIELD_WEIGHTS
) -> spmatrix:
    """
    Apply field weights to TF-IDF vectors and combine them with adaptive scaling
    
    Args:
        tfidf_data: Dictionary with TF-IDF vectors for each field
        field_weights: Dictionary mapping fields to weights
        
    Returns:
        Combined weighted document vectors
    """
    # Get shapes and create masks for empty documents
    title_mask = (tfidf_data['title_vectors'].getnnz(axis=1) > 0).astype(np.float64)
    abstract_mask = (tfidf_data['abstract_vectors'].getnnz(axis=1) > 0).astype(np.float64)
    body_mask = (tfidf_data['body_vectors'].getnnz(axis=1) > 0).astype(np.float64)
    topic_mask = (tfidf_data['topic_vectors'].getnnz(axis=1) > 0).astype(np.float64)
    
    # Check if we have keyword vectors
    has_keywords = 'keyword_vectors' in tfidf_data
    keyword_mask = np.zeros_like(title_mask)
    if has_keywords:
        keyword_mask = (tfidf_data['keyword_vectors'].getnnz(axis=1) > 0).astype(np.float64)
    
    # Calculate the combined weights for each document based on which fields exist
    total_weight = field_weights['title'] + field_weights['abstract'] + field_weights['body'] + field_weights['topics']
    if has_keywords and 'keywords' in field_weights:
        total_weight += field_weights['keywords']
    
    # Create a multiplier for each document that accounts for missing fields
    title_weight = field_weights['title'] * title_mask
    abstract_weight = field_weights['abstract'] * abstract_mask
    body_weight = field_weights['body'] * body_mask
    topic_weight = field_weights['topics'] * topic_mask
    keyword_weight = 0
    if has_keywords and 'keywords' in field_weights:
        keyword_weight = field_weights['keywords'] * keyword_mask
    
    # Calculate scaling factor to normalize weights
    doc_weights = title_weight + abstract_weight + body_weight + topic_weight
    if has_keywords:
        doc_weights += keyword_weight
    scaling_factors = np.divide(total_weight, doc_weights, out=np.zeros_like(doc_weights), where=doc_weights!=0)
    
    # Scale the weights by the scaling factors
    title_weight = np.multiply(title_weight, scaling_factors)
    abstract_weight = np.multiply(abstract_weight, scaling_factors)
    body_weight = np.multiply(body_weight, scaling_factors)
    topic_weight = np.multiply(topic_weight, scaling_factors)
    if has_keywords:
        keyword_weight = np.multiply(keyword_weight, scaling_factors)
    
    # Multiply each field by its weight and add them
    combined_vectors = (
        tfidf_data['title_vectors'].multiply(title_weight[:, np.newaxis]) +
        tfidf_data['abstract_vectors'].multiply(abstract_weight[:, np.newaxis]) +
        tfidf_data['body_vectors'].multiply(body_weight[:, np.newaxis]) +
        tfidf_data['topic_vectors'].multiply(topic_weight[:, np.newaxis])
    )
    
    # Add keyword vectors if available
    if has_keywords:
        combined_vectors = combined_vectors + tfidf_data['keyword_vectors'].multiply(keyword_weight[:, np.newaxis])
    
    logging.info("Applied adaptive field weighting and combined vectors")
    return combined_vectors

def apply_lsi(combined_vectors: spmatrix, n_dimensions: int = 150) -> Dict[str, Any]:
    """
    Apply Latent Semantic Indexing using SVD
    
    Args:
        combined_vectors: Combined and weighted document vectors
        n_dimensions: Number of LSI dimensions to use
        
    Returns:
        Dictionary with SVD model and transformed vectors
    """
    # Create SVD model for dimensionality reduction
    svd_model = TruncatedSVD(n_components=n_dimensions, random_state=42)
    
    # Apply SVD to create LSI representation
    lsi_vectors = svd_model.fit_transform(combined_vectors)
    
    # Normalize the vectors for cosine similarity
    normalized_lsi_vectors = normalize(lsi_vectors)
    
    logging.info(f"Applied LSI with {n_dimensions} dimensions")
    logging.info(f"Explained variance: {svd_model.explained_variance_ratio_.sum():.2%}")
    
    return {
        'svd_model': svd_model,
        'lsi_vectors': lsi_vectors,
        'normalized_lsi_vectors': normalized_lsi_vectors
    }

def save_model(model_data: Dict[str, Any], output_dir: str) -> None:
    """
    Save the model data to disk for later use
    
    Args:
        model_data: Dictionary with model components
        output_dir: Directory to save models
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the vectorizer
    joblib.dump(model_data['vectorizer'], os.path.join(output_dir, 'vectorizer.joblib'))
    logging.info(f"Saved vectorizer to {output_dir}/vectorizer.joblib")
    
    # Save the SVD model
    joblib.dump(model_data['svd_model'], os.path.join(output_dir, 'svd_model.joblib'))
    logging.info(f"Saved SVD model to {output_dir}/svd_model.joblib")
    
    # Save the normalized LSI vectors
    np.save(os.path.join(output_dir, 'lsi_vectors.npy'), model_data['normalized_lsi_vectors'])
    logging.info(f"Saved LSI vectors to {output_dir}/lsi_vectors.npy")
    
    # Save paper IDs
    joblib.dump(model_data['paper_ids'], os.path.join(output_dir, 'paper_ids.joblib'))
    logging.info(f"Saved paper IDs to {output_dir}/paper_ids.joblib")
    
    # Save field weights
    joblib.dump(model_data['field_weights'], os.path.join(output_dir, 'field_weights.joblib'))
    logging.info(f"Saved field weights to {output_dir}/field_weights.joblib")

def load_model(model_dir: str) -> Dict[str, Any]:
    """
    Load previously saved model from disk
    
    Args:
        model_dir: Directory containing saved model files
        
    Returns:
        Dictionary with loaded model components
    """
    vectorizer = joblib.load(os.path.join(model_dir, 'vectorizer.joblib'))
    logging.info(f"Loaded vectorizer from {model_dir}/vectorizer.joblib")
    
    svd_model = joblib.load(os.path.join(model_dir, 'svd_model.joblib'))
    logging.info(f"Loaded SVD model from {model_dir}/svd_model.joblib")
    
    normalized_lsi_vectors = np.load(os.path.join(model_dir, 'lsi_vectors.npy'))
    logging.info(f"Loaded LSI vectors from {model_dir}/lsi_vectors.npy")
    
    paper_ids = joblib.load(os.path.join(model_dir, 'paper_ids.joblib'))
    logging.info(f"Loaded paper IDs from {model_dir}/paper_ids.joblib")
    
    field_weights = joblib.load(os.path.join(model_dir, 'field_weights.joblib'))
    logging.info(f"Loaded field weights from {model_dir}/field_weights.joblib")
    
    return {
        'vectorizer': vectorizer,
        'svd_model': svd_model,
        'normalized_lsi_vectors': normalized_lsi_vectors,
        'paper_ids': paper_ids,
        'field_weights': field_weights
    }

def build_index(
    data_dir: str, 
    output_dir: str, 
    limit: Optional[int] = None,
    field_weights: Dict[str, float] = DEFAULT_FIELD_WEIGHTS,
    n_dimensions: int = 150,
    use_keybert: bool = False
) -> Dict[str, Any]:
    """
    Build a search index from JSON documents
    
    Args:
        data_dir: Directory containing JSON documents
        output_dir: Directory to save model files
        limit: Maximum number of documents to process
        field_weights: Weights to apply to document fields
        n_dimensions: Number of LSI dimensions
        use_keybert: Whether to use KeyBERT for keyword extraction
        
    Returns:
        Dictionary with model data
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load papers
    papers = load_papers(data_dir, limit=limit)
    
    # Extract fields
    extracted_fields = extract_fields(papers)
    
    # Extract keywords with KeyBERT if requested
    keyword_texts = None
    if use_keybert:
        try:
            keyword_texts = extract_keywords_with_keybert(extracted_fields)
        except Exception as e:
            logging.error(f"Error extracting keywords with KeyBERT: {e}")
            logging.warning("Continuing without KeyBERT keywords")
    
    # Create TF-IDF vectors
    tfidf_data = create_tfidf_vectors(extracted_fields, keyword_texts)
    
    # Apply field weighting
    combined_vectors = apply_field_weighting(tfidf_data, field_weights)
    
    # Apply LSI
    lsi_data = apply_lsi(combined_vectors, n_dimensions=n_dimensions)
    
    # Create the model data
    model_data = {
        'vectorizer': tfidf_data['vectorizer'],
        'svd_model': lsi_data['svd_model'],
        'lsi_vectors': lsi_data['lsi_vectors'],
        'normalized_lsi_vectors': lsi_data['normalized_lsi_vectors'],
        'paper_ids': extracted_fields['paper_ids'],
        'field_weights': field_weights,
        'use_keybert': use_keybert,
        'n_dimensions': n_dimensions
    }
    
    # Save the model
    save_model(model_data, output_dir)
    
    return model_data 
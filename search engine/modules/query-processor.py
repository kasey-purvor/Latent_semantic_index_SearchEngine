import logging
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from scipy.sparse import csr_matrix
import re
from pathlib import Path
import h5py

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("QueryProcessor")

class QueryProcessor:
    """
    Processes search queries and retrieves relevant documents.
    
    This module handles:
    1. Query normalization
    2. Projection into LSI space
    3. Similarity calculation
    4. Result boosting with enhancements
    5. Filtering by metadata
    """
    
    def __init__(self, 
                 indexing_engine,
                 keyword_extractor=None,
                 temporal_adjuster=None,
                 citation_calculator=None,
                 max_results: int = 100):
        """
        Initialize the query processor.
        
        Args:
            indexing_engine: LSI indexing engine
            keyword_extractor: Optional KeyBERT keyword extractor
            temporal_adjuster: Optional temporal relevance adjuster
            citation_calculator: Optional citation importance calculator
            max_results: Maximum number of results to return
        """
        self.indexing_engine = indexing_engine
        self.keyword_extractor = keyword_extractor
        self.temporal_adjuster = temporal_adjuster
        self.citation_calculator = citation_calculator
        self.max_results = max_results
        
        # Regular expressions for field queries
        self.field_pattern = re.compile(r'(\w+):"([^"]+)"')
        
        logger.info("QueryProcessor initialized")
    
    def normalize_query(self, query: str) -> Tuple[str, Dict[str, str]]:
        """
        Normalize a query and extract field filters.
        
        Args:
            query: Raw query string
            
        Returns:
            Tuple of (normalized query, field filters dictionary)
        """
        # Extract field filters
        field_filters = {}
        field_matches = self.field_pattern.findall(query)
        
        for field, value in field_matches:
            field_filters[field.lower()] = value
            # Remove the field filter from the query
            query = query.replace(f'{field}:"{value}"', '')
        
        # Normalize the remaining query
        normalized_query = query.lower()
        normalized_query = re.sub(r'\s+', ' ', normalized_query).strip()
        
        return normalized_query, field_filters
    
    def project_query_to_lsi(self, query: str) -> np.ndarray:
        """
        Project a query into LSI space.
        
        Args:
            query: Normalized query string
            
        Returns:
            Query vector in LSI space
        """
        # Create query TF-IDF representations for each field
        field_vectors = {}
        
        for field, vectorizer in self.indexing_engine.vectorizers.items():
            # Create a sparse vector for the query
            query_vector = vectorizer.transform([query])
            field_vectors[field] = query_vector * self.indexing_engine.field_weights[field]
        
        # Combine field vectors horizontally
        combined_vector = None
        for field, vector in field_vectors.items():
            if combined_vector is None:
                combined_vector = vector
            else:
                # This assumes the vectorizers have the same vocabulary order
                combined_vector = csr_matrix(np.hstack((combined_vector.toarray(), vector.toarray())))
        
        if combined_vector is None:
            # Create a zero vector if no fields matched
            vocab_size = sum(len(v.vocabulary_) for v in self.indexing_engine.vectorizers.values())
            combined_vector = csr_matrix((1, vocab_size))
        
        # Project into LSI space
        query_vector = self.indexing_engine.svd.transform(combined_vector)
        
        return query_vector
    
    def calculate_similarities(self, 
                              query_vector: np.ndarray, 
                              document_vectors: np.ndarray) -> np.ndarray:
        """
        Calculate cosine similarities between query and documents.
        
        Args:
            query_vector: Query vector in LSI space
            document_vectors: Document vectors in LSI space
            
        Returns:
            Array of similarity scores
        """
        # Normalize vectors for cosine similarity
        query_norm = np.linalg.norm(query_vector)
        doc_norms = np.linalg.norm(document_vectors, axis=1)
        
        # Avoid division by zero
        if query_norm == 0 or np.any(doc_norms == 0):
            # Return zeros for documents with zero norm
            similarities = np.zeros(document_vectors.shape[0])
            mask = doc_norms > 0
            if query_norm > 0 and np.any(mask):
                # Calculate similarities only for non-zero documents
                similarities[mask] = np.dot(document_vectors[mask], query_vector.T) / (doc_norms[mask] * query_norm)
        else:
            # Calculate cosine similarities
            similarities = np.dot(document_vectors, query_vector.T) / (doc_norms * query_norm)
        
        return similarities.flatten()
    
    def apply_boosting_factors(self, 
                              query: str, 
                              results: List[Dict[str, Any]], 
                              recency_preference: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Apply boosting factors from enhancement modules.
        
        Args:
            query: Normalized query string
            results: List of search result dictionaries
            recency_preference: Optional temporal recency preference
            
        Returns:
            List of search results with boosted scores
        """
        # Apply keyword boosting if extractor is available
        if self.keyword_extractor:
            for result in results:
                doc_id = result['id']
                keyword_boost = self.keyword_extractor.calculate_keyword_boost(query, doc_id)
                result['keyword_boost'] = keyword_boost
                result['score'] *= keyword_boost
        
        # Apply citation boosting if calculator is available
        if self.citation_calculator:
            for result in results:
                doc_id = result['id']
                citation_boost = self.citation_calculator.calculate_citation_boost(doc_id)
                result['citation_boost'] = citation_boost
                result['score'] *= citation_boost
        
        # Apply temporal boosting if adjuster is available
        if self.temporal_adjuster:
            results = self.temporal_adjuster.apply_temporal_boost(results, recency_preference)
        
        # Re-sort results by boosted scores
        results.sort(key=lambda x: x['score'], reverse=True)
        
        return results
    
    def filter_by_metadata(self, 
                         results: List[Dict[str, Any]], 
                         filters: Dict[str, str]) -> List[Dict[str, Any]]:
        """
        Filter search results by metadata fields.
        
        Args:
            results: List of search result dictionaries
            filters: Dictionary of field filters
            
        Returns:
            Filtered list of search results
        """
        if not filters:
            return results
        
        filtered_results = []
        
        for result in results:
            include = True
            
            for field, value in filters.items():
                if field == 'year' or field == 'date':
                    # Handle year ranges (e.g., "2010-2020")
                    if '-' in value:
                        start_year, end_year = value.split('-')
                        doc_year = result.get('year', 0)
                        if not (int(start_year) <= doc_year <= int(end_year)):
                            include = False
                            break
                    else:
                        # Exact year match
                        if str(result.get('year', '')) != value:
                            include = False
                            break
                
                elif field == 'author':
                    # Case-insensitive partial author match
                    authors = [author.lower() for author in result.get('authors', [])]
                    if not any(value.lower() in author for author in authors):
                        include = False
                        break
                
                elif field == 'journal' or field == 'venue':
                    # Case-insensitive partial journal match
                    journal = result.get('journal', '').lower()
                    if value.lower() not in journal:
                        include = False
                        break
                
                elif field == 'language':
                    # Exact language code match
                    if result.get('language', 'en') != value:
                        include = False
                        break
            
            if include:
                filtered_results.append(result)
        
        return filtered_results
    
    def expand_query(self, query: str, top_results: List[Dict[str, Any]], max_terms: int = 3) -> str:
        """
        Expand query with keywords from top results.
        
        Args:
            query: Original query
            top_results: List of top search results
            max_terms: Maximum number of terms to add
            
        Returns:
            Expanded query string
        """
        if not self.keyword_extractor or not top_results:
            return query
        
        # Extract document IDs from top results
        top_doc_ids = [result['id'] for result in top_results[:min(3, len(top_results))]]
        
        # Expand query using keyword extractor
        expanded_query = self.keyword_extractor.expand_query(query, top_doc_ids, max_terms)
        
        return expanded_query
    
    def search(self, 
              query: str, 
              recency_preference: Optional[float] = None,
              use_query_expansion: bool = False,
              num_results: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Search for documents matching a query.
        
        Args:
            query: Raw query string
            recency_preference: Optional temporal recency preference
            use_query_expansion: Whether to use query expansion
            num_results: Number of results to return (defaults to self.max_results)
            
        Returns:
            List of search results
        """
        if not query.strip():
            logger.warning("Empty query provided")
            return []
        
        # Normalize query and extract field filters
        normalized_query, field_filters = self.normalize_query(query)
        
        if not normalized_query and not field_filters:
            logger.warning("Query contains only stop words or is empty after normalization")
            return []
        
        # Load document vectors
        document_vectors, doc_ids = self.indexing_engine.load_document_vectors()
        
        if len(document_vectors) == 0:
            logger.error("No document vectors available for search")
            return []
        
        # Project query to LSI space
        query_vector = self.project_query_to_lsi(normalized_query)
        
        # Calculate similarities
        similarities = self.calculate_similarities(query_vector, document_vectors)
        
        # Create initial results list
        results = []
        for i, (doc_id, similarity) in enumerate(zip(doc_ids, similarities)):
            if similarity > 0:  # Only include documents with positive similarity
                results.append({
                    'id': doc_id,
                    'score': float(similarity),
                    'rank': i + 1
                })
        
        # Sort by similarity score
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # Apply query expansion if requested
        if use_query_expansion and self.keyword_extractor and results:
            expanded_query = self.expand_query(normalized_query, results[:3])
            
            if expanded_query != normalized_query:
                # Re-run search with expanded query
                query_vector = self.project_query_to_lsi(expanded_query)
                similarities = self.calculate_similarities(query_vector, document_vectors)
                
                # Update scores
                for i, (result, similarity) in enumerate(zip(results, similarities)):
                    result['score'] = float(similarity)
                    result['expanded_query'] = expanded_query
                
                # Re-sort by similarity score
                results.sort(key=lambda x: x['score'], reverse=True)
        
        # Apply boosting factors
        results = self.apply_boosting_factors(normalized_query, results, recency_preference)
        
        # Filter by metadata
        if field_filters:
            results = self.filter_by_metadata(results, field_filters)
        
        # Limit number of results
        max_results = num_results if num_results is not None else self.max_results
        results = results[:max_results]
        
        # Add rank information
        for i, result in enumerate(results):
            result['rank'] = i + 1
        
        logger.info(f"Query '{query}' returned {len(results)} results")
        return results

# Example usage:
if __name__ == "__main__":
    # This is a simple example to demonstrate how to use the QueryProcessor
    from indexing_engine import IndexingEngine
    from keyword_extractor import KeywordExtractor
    from temporal_relevance import TemporalRelevanceAdjuster
    
    # Initialize components
    indexing_engine = IndexingEngine(index_dir="./index")
    indexing_engine.load_index()
    
    keyword_extractor = KeywordExtractor(index_dir="./index")
    keyword_extractor.load_keywords()
    
    temporal_adjuster = TemporalRelevanceAdjuster(index_dir="./index")
    temporal_adjuster.load_years_index()
    
    # Initialize query processor
    processor = QueryProcessor(
        indexing_engine=indexing_engine,
        keyword_extractor=keyword_extractor,
        temporal_adjuster=temporal_adjuster
    )
    
    # Search example
    results = processor.search("machine learning neural networks", recency_preference=0.5)
    
    # Print results
    print(f"Found {len(results)} results:")
    for result in results[:5]:  # Show top 5
        print(f"Rank {result['rank']}: Document {result['id']} (Score: {result['score']:.4f})")
        if 'keyword_boost' in result:
            print(f"  Keyword Boost: {result['keyword_boost']:.2f}")
        if 'temporal_boost' in result:
            print(f"  Temporal Boost: {result['temporal_boost']:.2f}")

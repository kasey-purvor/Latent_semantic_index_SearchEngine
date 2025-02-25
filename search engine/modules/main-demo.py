import os
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any

# Import modules
from document_processor import DocumentProcessor
from indexing_engine import IndexingEngine
from keyword_extractor import KeywordExtractor
from temporal_relevance import TemporalRelevanceAdjuster
from query_processor import QueryProcessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AcademicSearchDemo")

class AcademicSearchEngine:
    """
    Academic paper search engine using LSI with enhancements.
    
    This class brings together all components of the search engine
    and provides a simple interface for indexing and searching.
    """
    
    def __init__(self, 
                 data_dir: str,
                 index_dir: str,
                 lsi_components: int = 150,
                 field_weights: Dict[str, float] = None,
                 batch_size: int = 1000):
        """
        Initialize the academic search engine.
        
        Args:
            data_dir: Directory containing JSON document files
            index_dir: Directory to store index files
            lsi_components: Number of LSI components
            field_weights: Dictionary mapping field names to weights
            batch_size: Size of document batches to process
        """
        self.data_dir = Path(data_dir)
        self.index_dir = Path(index_dir)
        
        # Create index directory if it doesn't exist
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize field weights
        self.field_weights = field_weights or {'title': 3.0, 'abstract': 1.5, 'full_text': 1.0}
        
        # Initialize components
        self.document_processor = DocumentProcessor(
            data_dir=data_dir,
            batch_size=batch_size
        )
        
        self.indexing_engine = IndexingEngine(
            index_dir=index_dir,
            n_components=lsi_components,
            field_weights=self.field_weights
        )
        
        self.keyword_extractor = KeywordExtractor(
            index_dir=index_dir,
            model_name="all-MiniLM-L6-v2",  # Lightweight model
            top_n=5
        )
        
        self.temporal_adjuster = TemporalRelevanceAdjuster(
            index_dir=index_dir,
            default_recency_preference=0.3
        )
        
        self.query_processor = QueryProcessor(
            indexing_engine=self.indexing_engine,
            keyword_extractor=self.keyword_extractor,
            temporal_adjuster=self.temporal_adjuster
        )
        
        logger.info(f"AcademicSearchEngine initialized with {lsi_components} LSI components")
        logger.info(f"Field weights: {self.field_weights}")
    
    def build_index(self, force_rebuild: bool = False):
        """
        Build the search index.
        
        Args:
            force_rebuild: Whether to force rebuilding the index
        """
        # Check if index already exists
        index_path = self.index_dir / "lsi_index.svd"
        if index_path.exists() and not force_rebuild:
            logger.info("Index already exists, loading from disk")
            self.load_index()
            return
        
        logger.info("Building index...")
        
        # Generate document batches
        batch_generator = self.document_processor.batch_document_generator()
        
        # Process first batch to initialize vectorizers
        try:
            first_batch = next(batch_generator)
        except StopIteration:
            logger.error("No documents found in data directory")
            return
        
        # Fit and transform first batch
        self.indexing_engine.fit_vectorizers(first_batch)
        doc_vectors, term_vectors = self.indexing_engine.fit_transform(first_batch)
        
        # Save document vectors
        doc_ids = [doc['id'] for doc in first_batch]
        self.indexing_engine.save_document_vectors(doc_vectors, doc_ids)
        
        # Extract keywords
        self.keyword_extractor.extract_keywords(first_batch)
        
        # Index document years
        self.temporal_adjuster.index_document_years(first_batch)
        
        # Process remaining batches
        batch_count = 1
        for batch in batch_generator:
            batch_count += 1
            logger.info(f"Processing batch {batch_count}")
            
            # Transform batch
            doc_vectors = self.indexing_engine.transform(batch)
            doc_ids = [doc['id'] for doc in batch]
            
            # Save vectors
            self.indexing_engine.save_document_vectors(doc_vectors, doc_ids)
            
            # Extract keywords
            self.keyword_extractor.extract_keywords(batch)
            
            # Index document years
            self.temporal_adjuster.index_document_years(batch)
        
        # Save all components
        self.indexing_engine.save_index()
        self.keyword_extractor.save_keywords()
        self.temporal_adjuster.save_years_index()
        
        logger.info(f"Index built successfully. Processed {batch_count} batches.")
    
    def load_index(self):
        """Load the search index from disk."""
        logger.info("Loading index components...")
        
        self.indexing_engine.load_index()
        self.keyword_extractor.load_keywords()
        self.temporal_adjuster.load_years_index()
        
        logger.info("Index loaded successfully")
    
    def search(self, 
              query: str, 
              recency_preference: float = None,
              use_query_expansion: bool = False,
              num_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search for documents matching a query.
        
        Args:
            query: Search query
            recency_preference: Preference for recent documents (0.0 to 1.0)
            use_query_expansion: Whether to use query expansion
            num_results: Number of results to return
            
        Returns:
            List of search results
        """
        results = self.query_processor.search(
            query=query,
            recency_preference=recency_preference,
            use_query_expansion=use_query_expansion,
            num_results=num_results
        )
        
        # Fetch document metadata for results
        for result in results:
            doc_id = result['id']
            
            # Add keywords
            if self.keyword_extractor:
                result['keywords'] = self.keyword_extractor.get_document_keywords(doc_id)
            
            # Add publication year
            if self.temporal_adjuster:
                result['year'] = self.temporal_adjuster.doc_years.get(doc_id)
        
        return results
    
    def fetch_document(self, doc_id: str) -> Dict[str, Any]:
        """
        Fetch a document by ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document metadata
        """
        # This is a simplified implementation that would need to be 
        # expanded in a real-world application to retrieve the actual document
        
        # In a real implementation, you would query your document store
        # For this demo, we'll return minimal metadata
        document = {"id": doc_id}
        
        # Add keywords
        if self.keyword_extractor:
            document['keywords'] = self.keyword_extractor.get_document_keywords(doc_id)
        
        # Add publication year
        if self.temporal_adjuster:
            document['year'] = self.temporal_adjuster.doc_years.get(doc_id)
        
        return document

def create_sample_data(output_dir: str, num_docs: int = 100):
    """
    Create sample documents for demonstration purposes.
    
    Args:
        output_dir: Directory to write sample documents
        num_docs: Number of documents to create
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Sample topics and related terms
    topics = {
        "machine_learning": [
            "neural networks", "deep learning", "supervised learning", 
            "classification", "regression", "clustering", "backpropagation"
        ],
        "information_retrieval": [
            "search engines", "indexing", "ranking", "relevance", 
            "query processing", "vector space model", "TF-IDF"
        ],
        "natural_language_processing": [
            "text mining", "sentiment analysis", "named entity recognition",
            "machine translation", "text classification", "word embeddings"
        ],
        "computer_vision": [
            "image recognition", "object detection", "segmentation",
            "convolutional neural networks", "facial recognition", "feature extraction"
        ]
    }
    
    # Sample authors
    authors = [
        ["Smith, J.", "Johnson, A."],
        ["Chen, L.", "Zhang, X.", "Wong, K."],
        ["Patel, N.", "Kumar, R."],
        ["Brown, T.", "Davis, M."],
        ["Garcia, C.", "Rodriguez, P."],
        ["Kim, S.", "Park, H."],
        ["Müller, J.", "Schmidt, K."],
        ["Taylor, E.", "Wilson, R."]
    ]
    
    # Sample journals
    journals = [
        "Journal of Machine Learning Research",
        "IEEE Transactions on Pattern Analysis and Machine Intelligence",
        "Information Retrieval Journal",
        "ACM Transactions on Information Systems",
        "Natural Language Engineering",
        "Computer Vision and Image Understanding",
        "Artificial Intelligence",
        "Data Mining and Knowledge Discovery"
    ]
    
    # Generate documents
    import random
    from datetime import datetime, timedelta
    
    for i in range(num_docs):
        # Select a random topic
        topic = random.choice(list(topics.keys()))
        related_terms = topics[topic]
        
        # Generate title
        title_terms = random.sample(related_terms, 2)
        title = f"Advances in {title_terms[0]} using {title_terms[1]}"
        
        # Generate abstract
        abstract_terms = random.sample(related_terms, min(4, len(related_terms)))
        abstract = f"This paper presents a novel approach to {abstract_terms[0]} and {abstract_terms[1]}. "
        abstract += f"We demonstrate how our method improves {abstract_terms[2]} by leveraging {abstract_terms[3]}. "
        abstract += f"Experimental results show significant improvements over baseline methods in the field of {topic.replace('_', ' ')}."
        
        # Generate metadata
        doc_authors = random.choice(authors)
        journal = random.choice(journals)
        
        # Generate year between 2000 and 2023
        year = random.randint(2000, 2023)
        
        # Create document ID
        doc_id = f"SAMPLE-{year}-{i+1:04d}"
        
        # Create document
        document = {
            "title": title,
            "abstract": abstract,
            "fullText": f"Full text content would be here for document {i+1}.",
            "year": year,
            "authors": doc_authors,
            "identifiers": [doc_id, f"10.1000/sample-{i+1}"],
            "journals": [{"title": journal, "identifiers": [f"issn:{random.randint(1000, 9999)}-{random.randint(1000, 9999)}"]}],
            "language": {"code": "en", "id": 9, "name": "English"}
        }
        
        # Write document to file
        with open(os.path.join(output_dir, f"doc_{i+1}.json"), 'w') as f:
            json.dump(document, f, indent=2)
    
    logger.info(f"Created {num_docs} sample documents in {output_dir}")

def demo_search_engine(data_dir: str, index_dir: str, query: str, recency_preference: float = 0.3):
    """
    Demonstrate the search engine functionality.
    
    Args:
        data_dir: Directory containing document data
        index_dir: Directory to store index files
        query: Search query to demonstrate
        recency_preference: Preference for recent documents (0.0 to 1.0)
    """
    # Initialize search engine
    search_engine = AcademicSearchEngine(
        data_dir=data_dir,
        index_dir=index_dir,
        lsi_components=50  # Reduced for demo purposes
    )
    
    # Build or load index
    try:
        search_engine.load_index()
        logger.info("Successfully loaded existing index")
    except Exception as e:
        logger.info(f"Could not load index: {str(e)}")
        logger.info("Building new index...")
        search_engine.build_index()
    
    # Perform search with default settings
    logger.info(f"\nSearch Query: '{query}'")
    logger.info("Default Search (No Enhancements):")
    results = search_engine.query_processor.search(
        query=query,
        use_query_expansion=False,
        recency_preference=None
    )
    
    # Display results
    print("\n=== Default Search Results ===")
    for i, result in enumerate(results[:5]):
        print(f"{i+1}. Document: {result['id']}")
        print(f"   Score: {result['score']:.4f}")
        print()
    
    # Perform search with keyword boosting only
    logger.info("\nSearch with Keyword Boosting:")
    # Temporarily disable temporal adjuster
    original_temporal_adjuster = search_engine.query_processor.temporal_adjuster
    search_engine.query_processor.temporal_adjuster = None
    
    results_with_keywords = search_engine.query_processor.search(
        query=query,
        use_query_expansion=False,
        recency_preference=None
    )
    
    # Restore temporal adjuster
    search_engine.query_processor.temporal_adjuster = original_temporal_adjuster
    
    # Display results
    print("\n=== Results with Keyword Boosting ===")
    for i, result in enumerate(results_with_keywords[:5]):
        print(f"{i+1}. Document: {result['id']}")
        print(f"   Score: {result['score']:.4f}")
        if 'keyword_boost' in result:
            print(f"   Keyword Boost: {result['keyword_boost']:.2f}")
        print()
    
    # Perform search with temporal boosting only
    logger.info("\nSearch with Temporal Boosting:")
    # Temporarily disable keyword extractor
    original_keyword_extractor = search_engine.query_processor.keyword_extractor
    search_engine.query_processor.keyword_extractor = None
    
    results_with_temporal = search_engine.query_processor.search(
        query=query,
        use_query_expansion=False,
        recency_preference=recency_preference
    )
    
    # Restore keyword extractor
    search_engine.query_processor.keyword_extractor = original_keyword_extractor
    
    # Display results
    print(f"\n=== Results with Temporal Boosting (Recency: {recency_preference}) ===")
    for i, result in enumerate(results_with_temporal[:5]):
        print(f"{i+1}. Document: {result['id']}")
        print(f"   Score: {result['score']:.4f}")
        if 'temporal_boost' in result:
            print(f"   Temporal Boost: {result['temporal_boost']:.2f}")
        if 'year' in result:
            print(f"   Year: {result['year']}")
        print()
    
    # Perform search with all enhancements
    logger.info("\nSearch with All Enhancements:")
    results_full = search_engine.search(
        query=query,
        use_query_expansion=True,
        recency_preference=recency_preference
    )
    
    # Display results
    print("\n=== Results with All Enhancements ===")
    for i, result in enumerate(results_full[:5]):
        print(f"{i+1}. Document: {result['id']}")
        print(f"   Score: {result['score']:.4f}")
        if 'keyword_boost' in result:
            print(f"   Keyword Boost: {result['keyword_boost']:.2f}")
        if 'temporal_boost' in result:
            print(f"   Temporal Boost: {result['temporal_boost']:.2f}")
        if 'keywords' in result:
            keywords_str = ", ".join([f"{kw[0]} ({kw[1]:.2f})" for kw in result['keywords'][:3]])
            print(f"   Top Keywords: {keywords_str}")
        if 'year' in result:
            print(f"   Year: {result['year']}")
        print()
    
    # Field Query Example
    field_query = f'{query} year:"2015-2023"'
    logger.info(f"\nSearch with Field Query: '{field_query}'")
    results_field = search_engine.search(
        query=field_query,
        use_query_expansion=True,
        recency_preference=recency_preference
    )
    
    # Display results
    print("\n=== Results with Field Filtering ===")
    for i, result in enumerate(results_field[:5]):
        print(f"{i+1}. Document: {result['id']}")
        print(f"   Score: {result['score']:.4f}")
        if 'year' in result:
            print(f"   Year: {result['year']}")
        print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Academic Search Engine Demo')
    parser.add_argument('--data_dir', type=str, default='./sample_data', help='Directory for document data')
    parser.add_argument('--index_dir', type=str, default='./index', help='Directory for index files')
    parser.add_argument('--query', type=str, default='neural networks deep learning', help='Search query')
    parser.add_argument('--recency', type=float, default=0.3, help='Recency preference (0.0-1.0)')
    parser.add_argument('--create_samples', action='store_true', help='Create sample documents')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of sample documents to create')
    
    args = parser.parse_args()
    
    # Create sample data if requested
    if args.create_samples:
        create_sample_data(args.data_dir, args.num_samples)
    
    # Run demo
    demo_search_engine(args.data_dir, args.index_dir, args.query, args.recency)

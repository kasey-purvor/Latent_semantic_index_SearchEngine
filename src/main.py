"""
Main application entry point for the Latent Semantic Search Engine.
Provides command-line interface for indexing and searching.
"""

import os
import sys
import argparse
import logging
from typing import List, Dict, Any, Optional

# Change from absolute to relative imports
from indexing import (
    load_papers,
    extract_fields,
    BaseIndexer
)
from search.query_processor import QueryProcessor
from faiss_bert_reranking import BertFaissReranker

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Default paths for data and models
DEFAULT_DATA_DIR = "../data/irCOREdata"
DEFAULT_MODEL_DIR = "../data/processed_data"
DEFAULT_OUTPUT_DIR = "../data/processed_data"

# Available index types
INDEX_TYPES = {
    'bm25': 'BM25 baseline',
    'lsi_basic': 'Basic LSI with k=150 dimensions',
    'lsi_field_weighted': 'Field-weighted LSI',
    'lsi_bert_enhanced': 'Field-weighted LSI with BERT-enhanced indexing'
}

def setup_argparse() -> argparse.ArgumentParser:
    """
    Set up command-line argument parser
    
    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description='Latent Semantic Search Engine',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Indexing command
    index_parser = subparsers.add_parser('index', help='Build search index from documents')
    index_parser.add_argument('--data-dir', default=DEFAULT_DATA_DIR, help='Directory containing JSON documents')
    index_parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR, help='Directory to save model files')
    index_parser.add_argument('--limit', type=int, help='Limit number of documents to process (for testing)')
    index_parser.add_argument('--dimensions', type=int, default=150, help='Number of LSI dimensions')
    index_parser.add_argument('--use-keybert', action='store_true', help='Use KeyBERT for keyword extraction (requires GPU for best performance)')
    index_parser.add_argument('--index-type', choices=list(INDEX_TYPES.keys()), 
                            default='lsi_field_weighted', help='Type of index to build')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search for documents')
    search_parser.add_argument('--model-dir', default=DEFAULT_MODEL_DIR, help='Directory containing model files')
    search_parser.add_argument('--method', choices=['binary', 'tfidf', 'log_entropy'], 
                              default='binary', help='Query representation method')
    search_parser.add_argument('--data-dir', default=DEFAULT_DATA_DIR, help='Directory with original JSON data (for result expansion)')
    search_parser.add_argument('--top-n', type=int, default=50, help='Number of results to return')
    search_parser.add_argument('--query', help='Search query (if not provided, interactive mode is used)')
    search_parser.add_argument('--index-type', choices=list(INDEX_TYPES.keys()), 
                            default='lsi_field_weighted', help='Type of index to use for search')
    search_parser.add_argument('--use-bert-reranking', action='store_true', 
                            help='Enable BERT-FAISS reranking for search results')
    search_parser.add_argument('--reranking-top-k', type=int, default=5, 
                            help='Number of results to return after BERT reranking')
    
    return parser

def handle_index_command(args: argparse.Namespace) -> None:
    """
    Handle the 'index' command
    
    Args:
        args: Command-line arguments
    """
    # Ensure data_dir and output_dir are set
    if not hasattr(args, 'data_dir') or not args.data_dir:
        args.data_dir = DEFAULT_DATA_DIR
    if not hasattr(args, 'output_dir') or not args.output_dir:
        args.output_dir = DEFAULT_OUTPUT_DIR
    if not hasattr(args, 'dimensions'):
        args.dimensions = 150
    if not hasattr(args, 'use_keybert'):
        args.use_keybert = (args.index_type == 'lsi_bert_enhanced')
        
    logging.info(f"Building {INDEX_TYPES[args.index_type]} index from documents in {args.data_dir}")
    
    if args.use_keybert:
        logging.info("KeyBERT keyword extraction enabled")
    
    # Load papers
    papers = load_papers(args.data_dir, limit=args.limit)
    
    # Extract fields
    extracted_data = extract_fields(papers)
    
    # Create output directory for this index type
    index_dir = os.path.join(args.output_dir, args.index_type)
    os.makedirs(index_dir, exist_ok=True)
    
    # Import and instantiate the appropriate indexer - using relative imports
    if args.index_type == 'bm25':
        from indexing.bm25 import BM25Indexer
        indexer = BM25Indexer()
    elif args.index_type == 'lsi_basic':
        from indexing.lsi import LSIIndexer
        indexer = LSIIndexer(n_components=args.dimensions)
    elif args.index_type == 'lsi_field_weighted':
        from indexing.field_weighted_lsi import FieldWeightedLSIIndexer
        indexer = FieldWeightedLSIIndexer(n_components=args.dimensions)
    elif args.index_type == 'lsi_bert_enhanced':
        from indexing.bert_enhanced import BertEnhancedIndexer
        indexer = BertEnhancedIndexer(n_components=args.dimensions)
    else:
        raise ValueError(f"Unknown index type: {args.index_type}")
    
    # Build the index
    indexer.build_index(extracted_data, index_dir)
    
    logging.info(f"Indexing complete. Model saved to {index_dir}")
    
    # If BERT reranking is enabled, prepare the BERT model and FAISS index
    if hasattr(args, 'use_bert_reranking') and args.use_bert_reranking:
        logging.info("Building BERT-FAISS index for reranking")
        
        # Ensure papers have paper_id for consistent matching with search results
        for paper in papers:
            if 'coreId' in paper and 'paper_id' not in paper:
                paper['paper_id'] = paper['coreId']
                
        bert_reranker = BertFaissReranker(model_name="C-KAI/sbert-academic-group44")
        bert_reranker.initialize()
        bert_reranker.build_index(papers, force_rebuild=True)

def handle_search_command(args: argparse.Namespace) -> None:
    """
    Handle the 'search' command
    
    Args:
        args: Command-line arguments
    """
    # Ensure model_dir and data_dir are set
    if not hasattr(args, 'model_dir') or not args.model_dir:
        args.model_dir = DEFAULT_MODEL_DIR
    if not hasattr(args, 'data_dir') or not args.data_dir:
        args.data_dir = DEFAULT_DATA_DIR
    if not hasattr(args, 'method'):
        args.method = 'binary'
    if not hasattr(args, 'top_n'):
        args.top_n = 50
    
    # Initialize BERT reranker if enabled
    bert_reranker = None
    if hasattr(args, 'use_bert_reranking') and args.use_bert_reranking:
        logging.info("Initializing BERT-FAISS reranker")
        bert_reranker = BertFaissReranker(model_name="C-KAI/sbert-academic-group44")
        bert_reranker.initialize()
        
    # Load the appropriate indexer. 
    if args.index_type == 'bm25':
        from indexing.bm25 import BM25Indexer
        indexer = BM25Indexer()
    elif args.index_type == 'lsi_basic':
        from indexing.lsi import LSIIndexer
        indexer = LSIIndexer()
    elif args.index_type == 'lsi_field_weighted':
        from indexing.field_weighted_lsi import FieldWeightedLSIIndexer
        indexer = FieldWeightedLSIIndexer()
    elif args.index_type == 'lsi_bert_enhanced':
        from indexing.bert_enhanced import BertEnhancedIndexer
        indexer = BertEnhancedIndexer()
    else:
        raise ValueError(f"Unknown index type: {args.index_type}")
    
    # Load the index
    index_dir = os.path.join(args.model_dir, args.index_type)
    logging.info(f"Loading {args.index_type} index from {index_dir}")
    indexer.load_index(index_dir)
    
    # Load original papers data if available (for result expansion)
    papers_data = []
    if args.data_dir:
        logging.info(f"Loading original papers data from {args.data_dir}")
        papers_data = load_papers(args.data_dir)
    
    # Interactive or single query mode
    if args.query:
        # Single query mode
        process_query(indexer, args.query, args.method, args.top_n, papers_data, bert_reranker, args.reranking_top_k if hasattr(args, 'reranking_top_k') else 5)
    else:
        # Interactive mode
        interactive_search(indexer, args.method, args.top_n, papers_data, bert_reranker, args.reranking_top_k if hasattr(args, 'reranking_top_k') else 5)

def process_query(
    indexer: BaseIndexer, 
    query: str, 
    method: str, 
    top_n: int,
    papers_data: List[Dict[str, Any]],
    bert_reranker: Optional[BertFaissReranker] = None,
    reranking_top_k: int = 5
) -> None:
    """
    Process a single search query
    
    Args:
        indexer: Initialized indexer instance
        query: Search query text
        method: Query representation method
        top_n: Number of results to return
        papers_data: Original papers data for result expansion
        bert_reranker: Optional BERT-FAISS reranker for reranking results
        reranking_top_k: Number of results to return after reranking
    """
    # Search for documents
    results = indexer.search(query, top_k=top_n)
    
    # Apply BERT reranking if enabled
    if bert_reranker is not None:
        logging.info("Applying BERT-FAISS reranking to search results")
        results = bert_reranker.rerank(query, results, top_k=reranking_top_k)
        logging.info(f"Reranking complete. Returning top {reranking_top_k} results.")
    
    # Display results
    print(f"\nSearch results for: '{query}' using {method} method")
    if bert_reranker is not None:
        print(f"Results reranked using BERT-FAISS")
    print(f"Found {len(results)} results")
    
    for i, result in enumerate(results):
        print(f"\n{i+1}. Score: {result['score']:.4f}")
        
        if bert_reranker is not None and 'bert_score' in result:
            print(f"   BERT Score: {result['bert_score']:.4f}")
        
        if 'title' in result:
            print(f"   Title: {result['title']}")
        
        if 'abstract' in result and result['abstract'] is not None:
            abstract = result['abstract']
            if len(abstract) > 200:
                abstract = abstract[:200] + "..."
            print(f"   Abstract: {abstract}")
        
        if 'topics' in result and result['topics']:
            topics_str = ", ".join(str(topic) for topic in result['topics'][:5])
            print(f"   Topics: {topics_str}")
        
        print(f"   Paper ID: {result['paper_id']}")
        
        if 'url' in result and result['url']:
            print(f"   URL: {result['url']}")

def interactive_search(
    indexer: BaseIndexer, 
    method: str, 
    top_n: int,
    papers_data: List[Dict[str, Any]],
    bert_reranker: Optional[BertFaissReranker] = None,
    reranking_top_k: int = 5
) -> None:
    """
    Run an interactive search loop
    
    Args:
        indexer: Initialized indexer instance
        method: Query representation method
        top_n: Number of results to return
        papers_data: Original papers data for result expansion
        bert_reranker: Optional BERT-FAISS reranker for reranking results
        reranking_top_k: Number of results to return after reranking
    """
    print(f"\nLatent Semantic Search Engine - Interactive Mode")
    print(f"Using {method} query representation method")
    print(f"Type 'quit' or 'exit' to end the session")
    print(f"Type 'method binary|tfidf|log_entropy' to change query method")
    print(f"Type 'top N' to change number of results (e.g., 'top 5')")
    if bert_reranker is not None:
        print(f"BERT-FAISS reranking is enabled (returning top {reranking_top_k} results)")
        print(f"Type 'rerank on|off' to enable/disable BERT reranking")
        print(f"Type 'rerank N' to change the number of reranked results")
    else:
        print(f"BERT-FAISS reranking is disabled")
        print(f"Type 'rerank on' to enable BERT reranking")
    print()
    
    current_method = method
    current_top_n = top_n
    use_reranker = bert_reranker is not None
    current_reranking_top_k = reranking_top_k
    
    # Initialize reranker if needed but not already initialized
    if use_reranker and bert_reranker.model is None:
        bert_reranker.initialize()
    
    while True:
        # Get query from user
        try:
            user_input = input("Enter your query: ").strip()
        except EOFError:
            print("\nExiting...")
            break
        except KeyboardInterrupt:
            print("\nExiting...")
            break
            
        if not user_input:
            continue
            
        if user_input.lower() in ['quit', 'exit']:
            print("\nExiting...")
            break
            
        # Check for method change
        if user_input.startswith('method '):
            method_name = user_input[7:].strip().lower()
            if method_name in ['binary', 'tfidf', 'log_entropy']:
                current_method = method_name
                print(f"Query method changed to {current_method}")
            else:
                print(f"Unknown method: {method_name}. Valid methods: binary, tfidf, log_entropy")
            continue
                
        # Check for top_n change
        if user_input.startswith('top '):
            try:
                new_top_n = int(user_input[4:].strip())
                if new_top_n > 0:
                    current_top_n = new_top_n
                    print(f"Number of results changed to {current_top_n}")
                else:
                    print("Number of results must be positive")
            except ValueError:
                print("Invalid number format")
            continue
        
        # Check for rerank command
        if user_input.startswith('rerank '):
            cmd = user_input[7:].strip().lower()
            
            if cmd == 'on':
                if bert_reranker is None:
                    # Initialize reranker if it doesn't exist
                    print("Initializing BERT-FAISS reranker...")
                    bert_reranker = BertFaissReranker()
                    bert_reranker.initialize()
                    
                    # Build the BERT index with papers data, forcing a rebuild
                    if papers_data:
                        print("Building BERT-FAISS index (this may take some time)...")
                        
                        # Ensure papers have paper_id for consistent matching with search results
                        for paper in papers_data:
                            if 'coreId' in paper and 'paper_id' not in paper:
                                paper['paper_id'] = paper['coreId']
                                
                        bert_reranker.build_index(papers_data, force_rebuild=True)
                    else:
                        print("Warning: No papers data available for building BERT index")
                use_reranker = True
                print(f"BERT reranking enabled (top {current_reranking_top_k} results)")
            elif cmd == 'off':
                use_reranker = False
                print("BERT reranking disabled")
            else:
                try:
                    new_k = int(cmd)
                    if new_k > 0:
                        current_reranking_top_k = new_k
                        print(f"Number of reranked results changed to {current_reranking_top_k}")
                    else:
                        print("Number of results must be positive")
                except ValueError:
                    print(f"Unknown rerank command: {cmd}. Valid commands: on, off, or a number")
            continue
                
        # Process search query
        process_query(
            indexer, 
            user_input, 
            current_method, 
            current_top_n, 
            papers_data,
            bert_reranker if use_reranker else None,
            current_reranking_top_k
        )

def prompt_for_command() -> argparse.Namespace:
    """
    Prompt the user for command input when no arguments are provided
    
    Returns:
        Namespace with user-provided arguments
    """
    parser = setup_argparse()
    
    print("\nLatent Semantic Search Engine - Interactive Setup")
    print("Please select a command:")
    print("1. index - Build search index from documents")
    print("2. search - Search for documents")
    choice = input("\nEnter your choice (1/2): ").strip()
    
    # Create a minimal set of arguments
    args = argparse.Namespace()
    
    # Set common defaults regardless of command
    args.data_dir = DEFAULT_DATA_DIR
    args.output_dir = DEFAULT_OUTPUT_DIR
    args.limit = None
    args.dimensions = 150
    
    if choice == '1':
        # Index command
        args.command = 'index'
        print("\nAvailable index types:")
        for idx, (index_type, description) in enumerate(INDEX_TYPES.items(), 1):
            print(f"{idx}. {description}")
        index_choice = input("\nSelect index type (1-4): ").strip()
        try:
            index_choice = int(index_choice)
            if 1 <= index_choice <= len(INDEX_TYPES):
                args.index_type = list(INDEX_TYPES.keys())[index_choice - 1]
            else:
                print("Invalid choice. Using default: lsi_field_weighted")
                args.index_type = 'lsi_field_weighted'
        except ValueError:
            print("Invalid input. Using default: lsi_field_weighted")
            args.index_type = 'lsi_field_weighted'
        
        # Ask for LSI dimensions if using an LSI-based index
        if 'lsi' in args.index_type:
            dimensions_input = input(f"\nEnter number of LSI dimensions (default: 150): ").strip()
            if dimensions_input:
                try:
                    args.dimensions = int(dimensions_input)
                    if args.dimensions <= 0:
                        print("Dimensions must be positive. Using default: 150")
                        args.dimensions = 150
                except ValueError:
                    print("Invalid input. Using default: 150")
                    args.dimensions = 150
        
        # Set use_keybert based on index type
        args.use_keybert = (args.index_type == 'lsi_bert_enhanced')
        
        # Ask if user wants to build BERT-FAISS index for reranking
        bert_reranking = input("\nBuild BERT-FAISS index for reranking? (y/n, default: n): ").strip().lower()
        args.use_bert_reranking = bert_reranking == 'y'
        
        if args.use_bert_reranking:
            # Always use the default model
            args.bert_model = "C-KAI/sbert-academic-group44"
            print(f"Using BERT model: {args.bert_model}")
            
    elif choice == '2':
        # Search command
        args.command = 'search'
        args.model_dir = DEFAULT_MODEL_DIR
        args.method = 'binary'
        args.top_n = 50
        args.query = None
        
        print("\nAvailable index types:")
        for idx, (index_type, description) in enumerate(INDEX_TYPES.items(), 1):
            print(f"{idx}. {description}")
        index_choice = input("\nSelect index type (1-4): ").strip()
        try:
            index_choice = int(index_choice)
            if 1 <= index_choice <= len(INDEX_TYPES):
                args.index_type = list(INDEX_TYPES.keys())[index_choice - 1]
            else:
                print("Invalid choice. Using default: lsi_field_weighted")
                args.index_type = 'lsi_field_weighted'
        except ValueError:
            print("Invalid input. Using default: lsi_field_weighted")
            args.index_type = 'lsi_field_weighted'
        
        # Ask about query representation method
        print("\nQuery representation methods:")
        print("1. binary - Simple term presence/absence")
        print("2. tfidf - Term frequency-inverse document frequency")
        print("3. log_entropy - Log entropy weighting")
        method_choice = input("Select query method (1-3, default: 1): ").strip()
        try:
            method_choice = int(method_choice)
            if method_choice == 1:
                args.method = 'binary'
            elif method_choice == 2:
                args.method = 'tfidf'
            elif method_choice == 3:
                args.method = 'log_entropy'
            else:
                print("Invalid choice. Using default: binary")
                args.method = 'binary'
        except ValueError:
            print("Invalid input. Using default: binary")
            args.method = 'binary'
        
        # Ask about number of results
        top_n_input = input("\nEnter number of results to return (default: 50): ").strip()
        if top_n_input:
            try:
                args.top_n = int(top_n_input)
                if args.top_n <= 0:
                    print("Number must be positive. Using default: 50")
                    args.top_n = 50
            except ValueError:
                print("Invalid input. Using default: 50")
                args.top_n = 50
        
        # Ask about BERT reranking
        bert_reranking = input("\nEnable BERT-FAISS reranking? (y/n, default: n): ").strip().lower()
        args.use_bert_reranking = bert_reranking == 'y'
        
        if args.use_bert_reranking:
            # Always use the default model
            args.bert_model = "C-KAI/sbert-academic-group44"
            print(f"Using BERT model: {args.bert_model}")
            
            # Ask about reranking result limit
            rerank_top_k_input = input("Enter number of results after reranking (default: 5): ").strip()
            if rerank_top_k_input:
                try:
                    args.reranking_top_k = int(rerank_top_k_input)
                    if args.reranking_top_k <= 0:
                        print("Number must be positive. Using default: 5")
                        args.reranking_top_k = 5
                except ValueError:
                    print("Invalid input. Using default: 5")
                    args.reranking_top_k = 5
            else:
                args.reranking_top_k = 5
            
    else:
        print("Invalid choice. Exiting.")
        sys.exit(1)
    
    return args

def main() -> None:
    """Main entry point"""
    parser = setup_argparse()
    args = parser.parse_args()
    
    if not args.command:
        args = prompt_for_command()
    
    if args.command == 'index':
        handle_index_command(args)
    elif args.command == 'search':
        handle_search_command(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main() 
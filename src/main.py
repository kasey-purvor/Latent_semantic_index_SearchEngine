"""
Main application entry point for the Latent Semantic Search Engine.
Provides command-line interface for indexing and searching.
"""

import os
import sys
import argparse
import logging
from typing import List, Dict, Any, Optional
import os.path as op

# Change from absolute to relative imports
from indexing import (
    load_papers,
    extract_fields,
    BaseIndexer
)
from indexing.bm25 import BM25Indexer
from search.query_processor import QueryProcessor
from faiss_bert_reranking import BertFaissReranker

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# New path construction using script location for absolute paths
SCRIPT_DIR = op.dirname(op.abspath(__file__))
PROJECT_ROOT = op.dirname(SCRIPT_DIR)
DEFAULT_DATA_DIR = op.join(PROJECT_ROOT, "data", "irCOREdata")
DEFAULT_MODEL_DIR = op.join(PROJECT_ROOT, "data", "processed_data")
DEFAULT_OUTPUT_DIR = op.join(PROJECT_ROOT, "data", "processed_data")

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
    
    # Add a command for downloading the BERT model for reranking
    bert_parser = subparsers.add_parser('download-bert-model', 
                                          help='Download and cache BERT model for search result reranking')
    bert_parser.add_argument('--bert-model', default="C-KAI/sbert-academic-group44", help='BERT model to use for reranking')
    
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
    search_parser.add_argument('--disable-bert-reranking', action='store_true', 
                            help='Disable BERT-FAISS reranking for search results (enabled by default)')
    search_parser.add_argument('--reranking-top-k', type=int, default=10, 
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
    
    # For BM25, always use binary method
    if args.index_type == 'bm25':
        args.method = 'binary'
        # Always disable BERT reranking for BM25
        args.disable_bert_reranking = True
        logging.info("Using binary query method for BM25 (only supported method)")
        logging.info("BERT reranking disabled for BM25 baseline")
    
    # Initialize BERT reranker by default unless explicitly disabled
    bert_reranker = None
    disable_bert_reranking = hasattr(args, 'disable_bert_reranking') and args.disable_bert_reranking
    
    if not disable_bert_reranking:
        logging.info("Initializing BERT reranker")
        bert_reranker = BertFaissReranker(model_name="C-KAI/sbert-academic-group44")
        try:
            bert_reranker.initialize()
        except Exception as e:
            logging.error(f"Failed to initialize BERT reranker: {e}")
            print(f"\nERROR: Failed to initialize BERT reranker: {e}")
            print("Please make sure you have the sentence-transformers package installed.")
            print("Continuing without BERT reranking.")
            bert_reranker = None
        
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
    try:
        indexer.load_index(index_dir)
    except Exception as e:
        logging.error(f"Failed to load index: {e}")
        print(f"\nERROR: Failed to load {args.index_type} index: {e}")
        print(f"Please build the index first with: python src/main.py index --index-type {args.index_type}")
        return
    
    # Load original papers data if available (for result expansion)
    papers_data = []
    if args.data_dir:
        logging.info(f"Loading original papers data from {args.data_dir}")
        papers_data = load_papers(args.data_dir)
    
    # Interactive or single query mode
    if args.query:
        # Single query mode
        process_query(indexer, args.query, args.method, args.top_n, papers_data, bert_reranker, args.reranking_top_k if hasattr(args, 'reranking_top_k') else 10)
    else:
        # Interactive mode
        interactive_search(indexer, args.method, args.top_n, papers_data, bert_reranker, args.reranking_top_k if hasattr(args, 'reranking_top_k') else 10)

def process_query(
    indexer: BaseIndexer, 
    query: str, 
    method: str, 
    top_n: int,
    papers_data: List[Dict[str, Any]],
    bert_reranker: Optional[BertFaissReranker] = None,
    reranking_top_k: int = 10
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
    logging.info(f"Search returned {len(results)} results initially")
    
    # Ensure all search results have string paper_id for consistent matching
    for result in results:
        if 'paper_id' in result and result['paper_id'] is not None:
            result['paper_id'] = str(result['paper_id'])
    
    # Skip BERT reranking for BM25 as it should remain a pure baseline
    is_bm25 = isinstance(indexer, BM25Indexer)
    
    # Apply BERT reranking if enabled and not BM25
    if bert_reranker is not None and not is_bm25:
        logging.info("Applying BERT-FAISS reranking to search results")
        results = bert_reranker.rerank(query, results, top_k=reranking_top_k)
        logging.info(f"Reranking complete. Returning top {reranking_top_k} results.")
    elif is_bm25:
        logging.info("Skipping BERT-FAISS reranking for BM25 baseline")
        # For BM25, only display the top results to match reranked count
        if len(results) > reranking_top_k:
            logging.info(f"BM25 found {len(results)} results, limiting to top {reranking_top_k}")
            results = results[:reranking_top_k]
            logging.info(f"Limiting BM25 results to top {reranking_top_k} for consistent display")
    
    # Display results
    print(f"\nSearch results for: '{query}' using {method} method")
    if bert_reranker is not None and not is_bm25:
        print(f"Results reranked using BERT-FAISS")
    elif is_bm25:
        if len(results) < top_n:
            print(f"Results from BM25 baseline (found {len(results)} documents containing query terms)")
        else:
            print(f"Results from BM25 baseline (top {min(len(results), reranking_top_k)} of {top_n} retrieved)")
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
    reranking_top_k: int = 10
) -> None:
    """
    Run interactive search loop
    
    Args:
        indexer: Initialized indexer instance
        method: Query representation method
        top_n: Number of results to return
        papers_data: Original papers data for result expansion
        bert_reranker: Optional BERT-FAISS reranker for reranking results
        reranking_top_k: Number of results to return after reranking
    """
    is_bm25 = isinstance(indexer, BM25Indexer)
    
    print(f"\nLatent Semantic Search Engine - Interactive Mode")
    
    if is_bm25:
        print(f"Using BM25 baseline with binary query representation")
    else:
        print(f"Using {method} query representation method")
        
    print(f"Type 'quit' or 'exit' to end the session")
    
    if not is_bm25:
        print(f"Type 'method binary|tfidf|log_entropy' to change query method")
        
    print(f"Type 'top N' to change number of results (e.g., 'top 5')")
    
    if is_bm25:
        print(f"BERT-FAISS reranking is disabled for BM25 baseline")
    elif bert_reranker is not None:
        print(f"BERT-FAISS reranking is enabled (returning top {reranking_top_k} results)")
        print(f"Type 'rerank off' to disable BERT reranking")
        print(f"Type 'rerank N' to change the number of reranked results")
    else:
        print(f"BERT-FAISS reranking is disabled")
        print(f"Note: BERT reranking is enabled by default, but failed to initialize")
        print(f"Please ensure you have installed the required packages and run 'download-bert-model'")
    print()
    
    current_method = method
    current_top_n = top_n
    use_reranker = bert_reranker is not None and not is_bm25
    current_reranking_top_k = reranking_top_k
    
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
            if is_bm25:
                print("Cannot change query method for BM25 - it only supports binary representation")
            else:
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
            if is_bm25:
                print("BERT reranking is disabled for BM25 baseline and cannot be enabled")
            else:
                cmd = user_input[7:].strip().lower()
                
                if cmd == 'on':
                    if bert_reranker is None:
                        print("ERROR: BERT-FAISS reranker is not initialized.")
                        print("Please restart with the --disable-bert-reranking flag and ensure")
                        print("the BERT-FAISS index is built using the 'download-bert-model' command.")
                    else:
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

def download_bert_model(papers: List[Dict[str, Any]] = None, limit: Optional[int] = None, 
                              model_name: str = "C-KAI/sbert-academic-group44", 
                              force_rebuild: bool = False) -> None:
    """
    Download and cache the BERT model for later use in search result reranking.
    This function doesn't build a traditional index - it just downloads and saves the model.
    
    Args:
        papers: List of paper documents (not used in this implementation)
        limit: Optional limit on number of papers to process (not used)
        model_name: Name of the BERT model to use
        force_rebuild: If True, force redownload of the model
    """
    logging.info(f"Downloading and caching BERT model '{model_name}' for reranking")
    print(f"\nDownloading and caching BERT model '{model_name}'...")
    print("This will be used for semantic reranking of search results.")
    
    # Initialize the BERT reranker
    bert_reranker = BertFaissReranker(model_name=model_name)
    bert_reranker.initialize()
    logging.info("BERT model downloaded and cached successfully")
    print("BERT model downloaded and cached successfully.")
    print("You can now use the --disable-bert-reranking flag with the search command.")

def handle_download_bert_model(args: argparse.Namespace) -> None:
    """
    Handle the 'download-bert-model' command
    
    Args:
        args: Command-line arguments
    """
    # Download the BERT model
    download_bert_model(
        model_name=args.bert_model if hasattr(args, 'bert_model') else "C-KAI/sbert-academic-group44"
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
    print("3. download-bert-model - Download and cache BERT model for search result reranking")
    choice = input("\nEnter your choice (1/2/3): ").strip()
    
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
        
        # The BERT reranking model is now initialized separately, so we don't need to ask here
        # Setting this to False to avoid redundancy
        args.build_bert_reranking_index = False
            
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
        
        # For BM25, always use binary method and disable BERT reranking
        if args.index_type == 'bm25':
            args.method = 'binary'
            args.disable_bert_reranking = True
            # Set standard number of results for BM25
            args.top_n = 50
            args.reranking_top_k = 10
            print("\nUsing binary query method for BM25 (only supported method)")
            print("BERT-FAISS reranking disabled for BM25 baseline")
            print(f"Using standard retrieval of 50 documents, displaying top 10 results")
        else:
            # Ask about query representation method (only for non-BM25 indexes)
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
        
            # Ask about number of results (only for non-BM25)
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
            
            # Ask about BERT reranking only for non-BM25 indexes
            bert_reranking = input("\nDisable BERT-FAISS reranking? (y/n, default: n): ").strip().lower()
            args.disable_bert_reranking = bert_reranking == 'y'
            
            if not args.disable_bert_reranking:
                # Always use the default model
                args.bert_model = "C-KAI/sbert-academic-group44"
                print(f"Using BERT model: {args.bert_model}")
                
                # Ask about reranking result limit
                rerank_top_k_input = input("Enter number of results after reranking (default: 10): ").strip()
                if rerank_top_k_input:
                    try:
                        args.reranking_top_k = int(rerank_top_k_input)
                        if args.reranking_top_k <= 0:
                            print("Number must be positive. Using default: 10")
                            args.reranking_top_k = 10
                    except ValueError:
                        print("Invalid input. Using default: 10")
                        args.reranking_top_k = 10
                else:
                    args.reranking_top_k = 10
    elif choice == '3':
        # Download BERT model command
        args.command = 'download-bert-model'
        args.limit = None
        args.force_rebuild = False
        args.bert_model = "C-KAI/sbert-academic-group44"
        
        # Ask if user wants to force rebuild
        force_rebuild = input("\nForce re-download of BERT model? (y/n, default: n): ").strip().lower()
        args.force_rebuild = force_rebuild == 'y'
        
        # Custom model option
        custom_model = input("\nUse custom BERT model? (Leave empty for default 'C-KAI/sbert-academic-group44'): ").strip()
        if custom_model:
            args.bert_model = custom_model
            print(f"Using custom BERT model: {args.bert_model}")
        else:
            print(f"Using default BERT model: {args.bert_model}")
    else:
        print("Invalid choice. Exiting.")
        sys.exit(1)
    
    return args

def main() -> None:
    """
    Main application entry point
    """
    # Process command-line arguments
    if len(sys.argv) > 1:
        # Command-line mode
        parser = setup_argparse()
        args = parser.parse_args()
        
        if args.command == 'index':
            handle_index_command(args)
        elif args.command == 'search':
            handle_search_command(args)
        elif args.command == 'download-bert-model':
            handle_download_bert_model(args)
        else:
            parser.print_help()
    else:
        # Interactive mode
        args = prompt_for_command()
        
        if args.command == 'index':
            handle_index_command(args)
        elif args.command == 'search':
            handle_search_command(args)
        elif args.command == 'download-bert-model':
            handle_download_bert_model(args)
        else:
            print("Invalid command. Exiting.")

if __name__ == '__main__':
    main() 
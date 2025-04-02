"""
Main application entry point for the Latent Semantic Search Engine.
Provides command-line interface for indexing and searching.
"""

import os
import sys
import argparse
import logging
from typing import List, Dict, Any, Optional

from indexing.core import (
    load_papers, 
    build_index, 
    load_model, 
    DEFAULT_FIELD_WEIGHTS
)
from search.query_processor import QueryProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Default paths for data and models
DEFAULT_DATA_DIR = "../data/irCOREdata"
DEFAULT_MODEL_DIR = "../data/processed_data"
DEFAULT_OUTPUT_DIR = "../data/processed_data"

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
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search for documents')
    search_parser.add_argument('--model-dir', default=DEFAULT_MODEL_DIR, help='Directory containing model files')
    search_parser.add_argument('--method', choices=['binary', 'tfidf', 'log_entropy'], 
                              default='binary', help='Query representation method')
    search_parser.add_argument('--data-dir', default=DEFAULT_DATA_DIR, help='Directory with original JSON data (for result expansion)')
    search_parser.add_argument('--top-n', type=int, default=10, help='Number of results to return')
    search_parser.add_argument('--query', help='Search query (if not provided, interactive mode is used)')
    
    return parser

def handle_index_command(args: argparse.Namespace) -> None:
    """
    Handle the 'index' command
    
    Args:
        args: Command-line arguments
    """
    logging.info(f"Building index from documents in {args.data_dir}")
    logging.info(f"Using {args.dimensions} LSI dimensions")
    
    if args.use_keybert:
        logging.info("KeyBERT keyword extraction enabled")
    
    # Build the index
    build_index(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        limit=args.limit,
        field_weights=DEFAULT_FIELD_WEIGHTS,
        n_dimensions=args.dimensions,
        use_keybert=args.use_keybert
    )
    
    logging.info(f"Indexing complete. Model saved to {args.output_dir}")

def handle_search_command(args: argparse.Namespace) -> None:
    """
    Handle the 'search' command
    
    Args:
        args: Command-line arguments
    """
    # Load the model
    logging.info(f"Loading model from {args.model_dir}")
    model_data = load_model(args.model_dir)
    
    # Initialize query processor
    query_processor = QueryProcessor(model_data)
    
    # Load original papers data if available (for result expansion)
    papers_data = []
    if args.data_dir:
        logging.info(f"Loading original papers data from {args.data_dir}")
        papers_data = load_papers(args.data_dir)
    
    # Interactive or single query mode
    if args.query:
        # Single query mode
        process_query(query_processor, args.query, args.method, args.top_n, papers_data)
    else:
        # Interactive mode
        interactive_search(query_processor, args.method, args.top_n, papers_data)

def process_query(
    query_processor: QueryProcessor, 
    query: str, 
    method: str, 
    top_n: int,
    papers_data: List[Dict[str, Any]]
) -> None:
    """
    Process a single search query
    
    Args:
        query_processor: Initialized QueryProcessor
        query: Search query text
        method: Query representation method
        top_n: Number of results to return
        papers_data: Original papers data for result expansion
    """
    # Search for documents
    results = query_processor.search(query, method=method, top_n=top_n)
    
    # Expand results if papers data is available
    if papers_data:
        results = query_processor.expand_results(results, papers_data)
    
    # Display results
    print(f"\nSearch results for: '{query}' using {method} method")
    print(f"Found {len(results)} results")
    
    for i, result in enumerate(results):
        print(f"\n{i+1}. Score: {result['score']:.4f}")
        
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
    query_processor: QueryProcessor, 
    method: str, 
    top_n: int,
    papers_data: List[Dict[str, Any]]
) -> None:
    """
    Run an interactive search loop
    
    Args:
        query_processor: Initialized QueryProcessor
        method: Query representation method
        top_n: Number of results to return
        papers_data: Original papers data for result expansion
    """
    print(f"\nLatent Semantic Search Engine - Interactive Mode")
    print(f"Using {method} query representation method")
    print(f"Type 'quit' or 'exit' to end the session")
    print(f"Type 'method binary|tfidf|log_entropy' to change query method")
    print(f"Type 'top N' to change number of results (e.g., 'top 5')\n")
    
    current_method = method
    current_top_n = top_n
    
    while True:
        try:
            query = input("\nEnter search query: ")
            query = query.strip()
            
            if query.lower() in ('quit', 'exit'):
                break
            
            # Check for method change command
            if query.lower().startswith('method '):
                parts = query.split()
                if len(parts) == 2 and parts[1] in ('binary', 'tfidf', 'log_entropy'):
                    current_method = parts[1]
                    print(f"Query method changed to: {current_method}")
                else:
                    print("Invalid method command. Use: method [binary|tfidf|log_entropy]")
                continue
            
            # Check for top_n change command
            if query.lower().startswith('top '):
                parts = query.split()
                if len(parts) == 2 and parts[1].isdigit():
                    current_top_n = int(parts[1])
                    print(f"Number of results changed to: {current_top_n}")
                else:
                    print("Invalid top command. Use: top [number]")
                continue
            
            if not query:
                continue
            
            # Process the query
            process_query(query_processor, query, current_method, current_top_n, papers_data)
            
        except KeyboardInterrupt:
            print("\nSearch session terminated by user.")
            break
        except Exception as e:
            print(f"Error processing query: {e}")

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
    
    if choice == '1':
        # Index command
        args.command = 'index'
        args.data_dir = DEFAULT_DATA_DIR
        args.output_dir = DEFAULT_OUTPUT_DIR
        
        limit_input = input("Enter document limit (optional, press Enter to skip): ").strip()
        args.limit = int(limit_input) if limit_input else None
        
        dimensions_input = input(f"Enter number of LSI dimensions (default: 150): ").strip()
        args.dimensions = int(dimensions_input) if dimensions_input else 150
        
        # Prompt for KeyBERT usage
        keybert_input = input("Use KeyBERT for keyword extraction? (y/n, default: n): ").strip().lower()
        args.use_keybert = keybert_input in ('y', 'yes', 'true')
        
        if args.use_keybert:
            print("KeyBERT keyword extraction enabled (requires GPU for best performance)")
        
    elif choice == '2':
        # Search command
        args.command = 'search'
        args.model_dir = DEFAULT_MODEL_DIR
        args.method = 'binary'
        args.data_dir = DEFAULT_DATA_DIR
        args.top_n = 10
        args.query = None
        
    else:
        print("Invalid choice. Exiting.")
        sys.exit(1)
    
    return args

def main() -> None:
    """
    Main application entry point
    """
    parser = setup_argparse()
    
    # Parse command-line arguments
    if len(sys.argv) > 1:
        args = parser.parse_args()
    else:
        # Interactive setup if no arguments are provided
        args = prompt_for_command()
    
    # Handle the selected command
    if args.command == 'index':
        handle_index_command(args)
    elif args.command == 'search':
        handle_search_command(args)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation terminated by user.")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Error: {e}")
        sys.exit(1) 
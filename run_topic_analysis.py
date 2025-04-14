#!/usr/bin/env python3
"""
Script to run topic distribution analysis on the academic paper corpus
"""

import os
import sys
import argparse

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from evaluation.test_query_generation.topic_distribution_analysis import TopicDistributionAnalyzer

def main():
    # Set up argument parser for command line options
    parser = argparse.ArgumentParser(description='Run topic distribution analysis')
    parser.add_argument('--data-dir', default='./data/irCOREdata', help='Directory containing paper data')
    parser.add_argument('--output-dir', default='./data/topic_analysis', help='Directory to save analysis results')
    parser.add_argument('--sample-size', type=int, default=None, help='Number of papers to sample (None to use all papers)')
    parser.add_argument('--num-topics', type=int, default=20, help='Number of topics to extract')
    parser.add_argument('--n-queries', type=int, default=50, help='Number of queries to generate')
    parser.add_argument('--use-full-text', action='store_true', help='Include full text in analysis')
    parser.add_argument('--ignore-incomplete', action='store_true', help='Filter out documents with missing fields')
    args = parser.parse_args()
    
    # Create the analyzer
    analyzer = TopicDistributionAnalyzer(
        output_dir=args.output_dir,
        num_topics=args.num_topics,
        use_full_text=args.use_full_text,
        ignore_incomplete_docs=args.ignore_incomplete
    )
    
    # Run the analysis
    print(f"Starting topic distribution analysis with {args.num_topics} topics")
    if args.sample_size:
        print(f"Using a sample of {args.sample_size} papers")
    else:
        print("Using all available papers")
    print(f"Using full text: {args.use_full_text}")
    print(f"Ignoring incomplete documents: {args.ignore_incomplete}")
    
    results = analyzer.run_analysis(args.data_dir, args.sample_size)
    
    print("\nAnalysis complete! Results saved to:", args.output_dir)
    
    # Print top topics
    print("\nTop 5 topics identified:")
    for i, topic_id in enumerate(results['top_topics'][:20]):
        keywords = ', '.join([word for word, _ in results['topic_keywords'][topic_id][:10]])
        proportion = results['overall_topic_distribution'][topic_id] * 100
        print(f"{i+1}. Topic {topic_id} ({proportion:.1f}%): {keywords}")
    
    # Print underrepresented topics
    if results['underrepresented_topics']:
        print("\nUnderrepresented topics:")
        for topic_id in results['underrepresented_topics']:
            keywords = ', '.join([word for word, _ in results['topic_keywords'][topic_id][:5]])
            print(f"Topic {topic_id}: {keywords}")
    
    # Generate queries
    queries = analyzer.generate_query_suggestions(results, n_queries=args.n_queries)
    
    # Print example queries
    print("\nExample queries for evaluation:")
    for i, query_data in enumerate(queries[:10]):
        topic_id = query_data['topic_id']
        query = query_data['query']
        print(f"{i+1}. \"{query}\" (Topic {topic_id})")
    
    print(f"\nFull list of {len(queries)} suggested queries saved to:", os.path.join(args.output_dir, "suggested_queries.csv"))

if __name__ == "__main__":
    main() 
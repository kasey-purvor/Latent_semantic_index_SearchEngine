"""
Topic Distribution Analysis using LDA for Academic Paper Corpus
Helps analyze topic distribution to generate balanced evaluation queries
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple, Optional
import logging
import random
from tqdm import tqdm
import pickle

# Gensim for LDA topic modeling
import gensim
from gensim import corpora
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel

# NLTK for text preprocessing
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Import functions from the indexing module
import sys
sys.path.append('../../')  # Add project root to path
from src.indexing.utils import load_papers, extract_fields

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Download necessary NLTK resources
def download_nltk_resources():
    """Download required NLTK resources"""
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')

class TopicDistributionAnalyzer:
    """Analyzes topic distribution in academic papers using LDA"""
    
    def __init__(self, 
                 output_dir: str = "./evaluation/test_query_generation/output",
                 num_topics: int = 100,
                 random_seed: int = 42,
                 use_full_text: bool = True,
                 ignore_incomplete_docs: bool = False):
        """
        Initialize the topic distribution analyzer
        
        Args:
            output_dir: Directory to save analysis results
            num_topics: Number of topics to extract with LDA
            random_seed: Random seed for reproducibility
            use_full_text: Whether to include full text in analysis
            ignore_incomplete_docs: Whether to filter out documents with missing fields
        """
        self.output_dir = output_dir
        self.num_topics = num_topics
        self.random_seed = random_seed
        self.use_full_text = use_full_text
        self.ignore_incomplete_docs = ignore_incomplete_docs
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize containers
        self.papers = None
        self.corpus = None
        self.dictionary = None
        self.lda_model = None
        self.doc_topic_distribution = None
    
    def load_and_sample_papers(self, 
                              data_dir: str, 
                              sample_size: int = None) -> List[Dict[str, Any]]:
        """
        Load and sample papers from the corpus
        
        Args:
            data_dir: Directory containing the paper data
            sample_size: Number of papers to sample (None to use all papers)
            
        Returns:
            List of sampled paper dictionaries
        """
        # Load all papers
        logging.info(f"Loading papers from {data_dir}")
        all_papers = load_papers(data_dir)
        
        if self.ignore_incomplete_docs:
            # Filter papers with complete fields
            if self.use_full_text:
                # Require all fields if using full text
                filtered_papers = [p for p in all_papers if 
                                p.get('title') and 
                                p.get('abstract') and 
                                p.get('fullText')]
                logging.info(f"Found {len(filtered_papers)} papers with all fields (title, abstract, fullText) "
                           f"out of {len(all_papers)} total papers")
            else:
                # Require at least title or abstract if not using full text
                filtered_papers = [p for p in all_papers if 
                                  (p.get('title') or p.get('abstract'))]
                logging.info(f"Found {len(filtered_papers)} papers with at least title or abstract "
                           f"out of {len(all_papers)} total papers")
        else:
            # Use all papers, even those with missing fields
            filtered_papers = all_papers
            logging.info(f"Using all {len(all_papers)} papers, including those with missing fields")
        
        # Sample papers if needed and if a sample size is specified
        if sample_size is not None and sample_size < len(filtered_papers):
            random.seed(self.random_seed)
            sampled_papers = random.sample(filtered_papers, sample_size)
            logging.info(f"Sampled {sample_size} papers for analysis")
        else:
            sampled_papers = filtered_papers
            logging.info(f"Using all {len(filtered_papers)} filtered papers for analysis")
        
        self.papers = sampled_papers
        return sampled_papers
    
    def preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text for topic modeling
        
        Args:
            text: Text to preprocess
            
        Returns:
            List of preprocessed tokens
        """
        if not isinstance(text, str) or not text:
            return []
            
        # Lower case
        text = text.lower()
        
        try:
            # Tokenize - use a simpler approach to avoid potential issues
            tokens = text.split()
            
            # Remove stopwords, punctuation, and short words
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                     if token not in self.stop_words 
                     and token.isalpha() 
                     and len(token) > 2]
            
            return tokens
        except Exception as e:
            logging.error(f"Error preprocessing text: {e}")
            return []
    
    def extract_paper_content(self, papers: List[Dict[str, Any]]) -> List[str]:
        """
        Extract and combine content from papers based on configuration
        
        Args:
            papers: List of paper dictionaries
            
        Returns:
            List of combined paper texts
        """
        texts = []
        for paper in papers:
            title = paper.get('title', '')
            abstract = paper.get('abstract', '')
            
            # Combine available fields based on configuration
            if self.use_full_text:
                full_text = paper.get('fullText', '')
                combined_text = f"{title} {abstract} {full_text}"
            else:
                combined_text = f"{title} {abstract}"
            
            texts.append(combined_text.strip())
        
        return texts
    
    def build_lda_model(self, 
                       papers: Optional[List[Dict[str, Any]]] = None, 
                       texts: Optional[List[str]] = None) -> Tuple[LdaModel, List[List[Tuple[int, float]]]]:
        """
        Build LDA topic model from papers
        
        Args:
            papers: List of paper dictionaries (optional)
            texts: List of preprocessed texts (optional)
            
        Returns:
            Tuple of (LDA model, corpus)
        """
        if papers is None:
            papers = self.papers
        
        if texts is None:
            logging.info("Extracting text from papers")
            texts = self.extract_paper_content(papers)
        
        # Preprocess texts
        logging.info("Preprocessing texts")
        processed_texts = [self.preprocess_text(text) for text in tqdm(texts)]
        
        # Create dictionary
        logging.info("Creating dictionary")
        dictionary = corpora.Dictionary(processed_texts)
        
        # Filter extremes (optional)
        dictionary.filter_extremes(no_below=5, no_above=0.7)
        
        # Create corpus
        logging.info("Creating corpus")
        corpus = [dictionary.doc2bow(text) for text in processed_texts]
        
        # Build LDA model
        logging.info(f"Building LDA model with {self.num_topics} topics")
        lda_model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=self.num_topics,
            random_state=self.random_seed,
            passes=10,
            alpha='auto',
            per_word_topics=True
        )
        
        # Store the model and corpus
        self.corpus = corpus
        self.dictionary = dictionary
        self.lda_model = lda_model
        
        return lda_model, corpus
    
    def compute_coherence_score(self) -> float:
        """
        Compute coherence score for the LDA model
        
        Returns:
            Coherence score
        """
        coherence_model = CoherenceModel(
            model=self.lda_model,
            texts=self.corpus,
            dictionary=self.dictionary,
            coherence='c_v'
        )
        
        coherence_score = coherence_model.get_coherence()
        logging.info(f"Topic model coherence score: {coherence_score:.4f}")
        
        return coherence_score
    
    def get_document_topics(self) -> np.ndarray:
        """
        Get topic distribution for each document
        
        Returns:
            Array of topic distributions for documents
        """
        if self.doc_topic_distribution is not None:
            return self.doc_topic_distribution
        
        logging.info("Computing document-topic distribution")
        
        # Get document-topic distribution
        doc_topics = []
        for doc in self.corpus:
            topic_dist = self.lda_model.get_document_topics(doc)
            # Convert to full distribution vector
            topic_vector = np.zeros(self.num_topics)
            for topic_id, prob in topic_dist:
                topic_vector[topic_id] = prob
            doc_topics.append(topic_vector)
        
        self.doc_topic_distribution = np.array(doc_topics)
        return self.doc_topic_distribution
    
    def analyze_topic_distribution(self) -> Dict[str, Any]:
        """
        Analyze the distribution of topics in the corpus
        
        Returns:
            Dictionary with topic distribution analysis
        """
        # Get document-topic distribution
        doc_topics = self.get_document_topics()
        
        # Calculate overall topic distribution in corpus
        overall_topic_dist = doc_topics.mean(axis=0)
        
        # Get top topics
        top_topics_idx = np.argsort(-overall_topic_dist)
        
        # Get topic keywords
        topic_keywords = {}
        for topic_id in range(self.num_topics):
            keywords = self.lda_model.show_topic(topic_id, topn=10)
            topic_keywords[topic_id] = keywords
        
        # Calculate dominant topic for each document
        dominant_topics = np.argmax(doc_topics, axis=1)
        dominant_topic_counts = np.bincount(dominant_topics, minlength=self.num_topics)
        dominant_topic_percent = dominant_topic_counts / len(dominant_topics)
        
        # Identify under and over-represented topics
        mean_representation = 1.0 / self.num_topics
        underrepresented = [topic_id for topic_id in range(self.num_topics) 
                          if overall_topic_dist[topic_id] < 0.5 * mean_representation]
        
        overrepresented = [topic_id for topic_id in range(self.num_topics) 
                         if overall_topic_dist[topic_id] > 2 * mean_representation]
        
        # Prepare results
        results = {
            'overall_topic_distribution': overall_topic_dist,
            'top_topics': top_topics_idx,
            'topic_keywords': topic_keywords,
            'dominant_topic_counts': dominant_topic_counts,
            'dominant_topic_percent': dominant_topic_percent,
            'underrepresented_topics': underrepresented,
            'overrepresented_topics': overrepresented
        }
        
        return results
    
    def visualize_topic_distribution(self, results: Dict[str, Any]) -> None:
        """
        Visualize topic distribution analysis
        
        Args:
            results: Results from topic distribution analysis
        """
        # Set up the figure
        plt.figure(figsize=(12, 10))
        
        # Create a DataFrame for the topic distribution
        topic_data = []
        for topic_id in range(self.num_topics):
            keywords = ', '.join([word for word, _ in results['topic_keywords'][topic_id][:10]])
            topic_data.append({
                'Topic': f'Topic {topic_id}',
                'Keywords': keywords,
                'Proportion': results['overall_topic_distribution'][topic_id],
                'Document Count': results['dominant_topic_counts'][topic_id]
            })
        
        df_topics = pd.DataFrame(topic_data)
        df_topics = df_topics.sort_values('Proportion', ascending=False)
        
        # Plot topic distribution
        plt.subplot(2, 1, 1)
        sns.barplot(x='Topic', y='Proportion', data=df_topics)
        plt.title('Topic Distribution in Corpus')
        plt.xticks(rotation=90)
        plt.tight_layout()
        
        # Plot document count by dominant topic
        plt.subplot(2, 1, 2)
        sns.barplot(x='Topic', y='Document Count', data=df_topics)
        plt.title('Document Count by Dominant Topic')
        plt.xticks(rotation=90)
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(os.path.join(self.output_dir, 'topic_distribution.png'))
        logging.info(f"Saved topic distribution visualization to {self.output_dir}/topic_distribution.png")
        
        # Create a heatmap of topic keywords
        plt.figure(figsize=(14, 10))
        topic_keyword_matrix = np.zeros((self.num_topics, 10))
        keyword_labels = []
        
        # Get all keywords
        for topic_id in range(self.num_topics):
            for i, (keyword, weight) in enumerate(results['topic_keywords'][topic_id][:10]):
                topic_keyword_matrix[topic_id, i] = weight
                if i >= len(keyword_labels):
                    keyword_labels.append(keyword)
        
        # Create heatmap
        sns.heatmap(topic_keyword_matrix, cmap="YlGnBu", 
                   xticklabels=keyword_labels, 
                   yticklabels=[f"Topic {i}" for i in range(self.num_topics)])
        plt.title('Topic-Keyword Heatmap')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'topic_keyword_heatmap.png'))
        logging.info(f"Saved topic keyword heatmap to {self.output_dir}/topic_keyword_heatmap.png")
    
    def generate_query_suggestions(self, results: Dict[str, Any], n_queries: int = 20) -> List[str]:
        """
        Generate query suggestions based on topic distribution
        
        Args:
            results: Results from topic distribution analysis
            n_queries: Number of query suggestions to generate
            
        Returns:
            List of query suggestions
        """
        queries = []
        
        # Get topic distribution to determine how many queries to generate per topic
        topic_dist = results['overall_topic_distribution']
        
        # Normalize to ensure we generate exactly n_queries
        topic_query_counts = np.round(topic_dist * n_queries).astype(int)
        
        # Adjust to ensure we get exactly n_queries
        while sum(topic_query_counts) < n_queries:
            # Add to the topics with highest distribution first
            idx = np.argmax(topic_dist - topic_query_counts / n_queries)
            topic_query_counts[idx] += 1
        
        while sum(topic_query_counts) > n_queries:
            # Remove from the topics with lowest distribution first
            non_zero_idx = np.where(topic_query_counts > 0)[0]
            idx = non_zero_idx[np.argmin(topic_dist[non_zero_idx])]
            topic_query_counts[idx] -= 1
        
        # Generate queries for each topic based on its keywords
        for topic_id in range(self.num_topics):
            n_topic_queries = topic_query_counts[topic_id]
            if n_topic_queries == 0:
                continue
                
            # Get keywords for this topic
            keywords = results['topic_keywords'][topic_id]
            
            # Generate queries by combining keywords
            for i in range(n_topic_queries):
                # Select 2-3 keywords from top 7
                n_keywords = random.randint(2, 3)
                selected_keywords = random.sample([kw for kw, _ in keywords[:7]], n_keywords)
                
                # Combine into a query
                query = ' '.join(selected_keywords)
                queries.append({
                    'query': query,
                    'topic_id': topic_id,
                    'keywords': selected_keywords
                })
        
        # Save queries to file
        query_df = pd.DataFrame(queries)
        query_df.to_csv(os.path.join(self.output_dir, 'suggested_queries.csv'), index=False)
        logging.info(f"Saved {len(queries)} query suggestions to {self.output_dir}/suggested_queries.csv")
        
        return queries
    
    def save_model_and_results(self, results: Dict[str, Any]) -> None:
        """
        Save model and analysis results
        
        Args:
            results: Results from topic distribution analysis
        """
        # Save LDA model
        self.lda_model.save(os.path.join(self.output_dir, 'lda_model'))
        
        # Save dictionary
        self.dictionary.save(os.path.join(self.output_dir, 'dictionary'))
        
        # Save document-topic distribution
        with open(os.path.join(self.output_dir, 'doc_topic_distribution.pkl'), 'wb') as f:
            pickle.dump(self.doc_topic_distribution, f)
        
        # Save analysis results
        with open(os.path.join(self.output_dir, 'topic_analysis_results.pkl'), 'wb') as f:
            pickle.dump(results, f)
        
        # Save topic keywords as text file
        with open(os.path.join(self.output_dir, 'topic_keywords.txt'), 'w') as f:
            for topic_id in range(self.num_topics):
                keywords = results['topic_keywords'][topic_id]
                keyword_str = ', '.join([f"{word} ({weight:.4f})" for word, weight in keywords])
                f.write(f"Topic {topic_id}: {keyword_str}\n\n")
        
        # Save metadata about the analysis
        with open(os.path.join(self.output_dir, 'analysis_metadata.txt'), 'w') as f:
            f.write(f"Number of documents: {len(self.papers)}\n")
            f.write(f"Number of topics: {self.num_topics}\n")
            f.write(f"Used full text: {self.use_full_text}\n")
            f.write(f"Ignored incomplete documents: {self.ignore_incomplete_docs}\n")
        
        logging.info(f"Saved model and results to {self.output_dir}")
    
    def run_analysis(self, data_dir: str, sample_size: int = None) -> Dict[str, Any]:
        """
        Run complete topic distribution analysis
        
        Args:
            data_dir: Directory containing the paper data
            sample_size: Number of papers to sample (None to use all papers)
            
        Returns:
            Dictionary with topic distribution analysis results
        """
        # Download NLTK resources
        download_nltk_resources()
        
        # Load and sample papers
        self.load_and_sample_papers(data_dir, sample_size)
        
        # Build LDA model
        self.build_lda_model()
        
        # Compute coherence score
        self.compute_coherence_score()
        
        # Analyze topic distribution
        results = self.analyze_topic_distribution()
        
        # Visualize topic distribution
        self.visualize_topic_distribution(results)
        
        # Generate query suggestions
        self.generate_query_suggestions(results)
        
        # Save model and results
        self.save_model_and_results(results)
        
        return results

def main():
    """Main function to run topic distribution analysis"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Topic Distribution Analysis')
    parser.add_argument('--data-dir', default='./data/irCOREdata', help='Directory containing paper data')
    parser.add_argument('--output-dir', default='./evaluation/test_query_generation/output', help='Directory to save analysis results')
    parser.add_argument('--sample-size', type=int, default=None, help='Number of papers to sample (None to use all papers)')
    parser.add_argument('--num-topics', type=int, default=20, help='Number of topics to extract')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--use-full-text', action='store_true', help='Include full text in analysis')
    parser.add_argument('--ignore-incomplete', action='store_true', help='Filter out documents with missing fields')
    
    args = parser.parse_args()
    
    analyzer = TopicDistributionAnalyzer(
        output_dir=args.output_dir,
        num_topics=args.num_topics,
        random_seed=args.seed,
        use_full_text=args.use_full_text,
        ignore_incomplete_docs=args.ignore_incomplete
    )
    
    analyzer.run_analysis(args.data_dir, args.sample_size)

if __name__ == '__main__':
    main() 
# Latent Semantic Index Search Engine
A search engine providing semantic search ability over academic papers.

## Features

- Multiple search index types (BM25, LSI basic, LSI field-weighted, BERT-enhanced LSI)
- Command-line interface for indexing and searching
- Interactive search mode
- BERT-FAISS reranking for improved search quality

## Installation

### Method 1: Using requirements.txt
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`

### Method 2: Manual installation
Install core dependencies:
```bash
pip install numpy scipy scikit-learn pandas joblib tqdm matplotlib seaborn gensim nltk
```

Install BERT and FAISS for semantic reranking:
```bash
pip install torch sentence-transformers faiss-cpu
```

Install visualization and keyword extraction tools:
```bash
pip install pyLDAvis keybert
```

## Usage

### Indexing documents

```bash
python src/main.py index --data-dir path/to/documents --index-type bm25
```

Available index types:
- `bm25`: BM25 baseline
- `lsi_basic`: Basic LSI with default 150 dimensions
- `lsi_field_weighted`: Field-weighted LSI
- `lsi_bert_enhanced`: Field-weighted LSI with BERT-enhanced indexing

### Searching

```bash
# Basic search
python src/main.py search --index-type bm25

# Search with BERT-FAISS reranking
python src/main.py search --index-type bm25 --use-bert-reranking
```

Search options:
- `--model-dir`: Directory containing model files (default: ../data/processed_data)
- `--method`: Query representation method (binary, tfidf, log_entropy)
- `--top-n`: Number of results to return (default: 50)
- `--query`: Search query (if not provided, interactive mode is used)
- `--use-bert-reranking`: Enable BERT-FAISS reranking for search results
- `--reranking-top-k`: Number of results to return after BERT reranking (default: 5)

### Interactive mode

You can also run the program without arguments to enter interactive setup mode:
```bash
python src/main.py
```

This will guide you through setting up your search or indexing operation with prompts.

In interactive search mode, the following commands are available:
- `method binary|tfidf|log_entropy`: Change query representation method
- `top N`: Change number of results (e.g., `top 5`)
- `rerank on|off`: Enable/disable BERT-FAISS reranking
- `rerank N`: Change number of reranked results (e.g., `rerank 10`)
- `quit` or `exit`: Exit the program

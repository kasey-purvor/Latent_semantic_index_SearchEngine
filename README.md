# Latent Semantic Index Search Engine
A search engine providing semantic search ability over academic papers.

## Features

- Browser-based frontend for interactive searching
- Multiple search index types (BM25, LSI basic, LSI field-weighted, BERT-enhanced LSI)
- Command-line interface for indexing, searching and evaluation
- BERT-FAISS reranking for improved search quality (with GPU acceleration)

## Installation

### Environment Setup

The easiest way to set up the environment is using Conda with the provided environment.yml file:

```bash
# Create and activate the environment
conda env create -f environment.yml
conda activate search_engine

# Install frontend dependencies
cd frontend
npm install
cd ..
```

### PyTorch Installation

#### For Windows
```bash
# Install PyTorch with CUDA support
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

**Note:** BERT models use GPU acceleration, Windows will require Visual Studio with C++ extensions and runtime libraries.

#### For Linux
```bash
# Install PyTorch with CUDA support
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

#### For Mac
```bash
# Install PyTorch (Mac does not support CUDA)
pip3 install torch torchvision torchaudio
```

**Note:** Mac does not support CUDA, so BERT-enhanced models will run on CPU only, which may be significantly slower.

## Usage

### Running the Full Application (Frontend and Backend)

The easiest way to run the complete application is to use the provided controller script:

```bash
# Run both frontend and backend
python run.py
```

This will start:
- The Flask backend API on port 5000
- The React frontend on port 5173 (may be different if the port is already in use)

You can then access the application by opening your browser to http://localhost:5173

To stop both services, press Ctrl+C in the terminal where you started them.

### Indexing Documents and Model Setup

**Note for Dropbox users:** If you received this application via Dropbox, the indexes are already built and the BERT model (and the judgeblender LLMs) are already downloaded. You can skip the rest of this instructions and go directly to running the application ```bash
# Run both frontend and backend
python run.py
```

**Only if cloning from GitHub:** You'll need to build the search indexes and download the BERT model(and the judgeblender LLMs) before using the application.

To build the indexes:

```bash
python src/main.py index
```

This will start an interactive setup that guides you through the indexing process.

### Download BERT Model for Reranking

**Only if cloning from GitHub:** To use the BERT-FAISS reranking feature, you need to download the model:

```bash
python src/main.py download-bert-model
```

This will download the necessary BERT model from Hugging Face for improved search result ranking.

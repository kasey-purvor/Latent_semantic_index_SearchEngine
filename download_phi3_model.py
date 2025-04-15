#!/usr/bin/env python3
"""
Script to download Phi-3-mini model from Hugging Face and save it locally.
This script should be run from the project root directory.
"""

import os
import sys
from pathlib import Path
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError:
    logging.error("The transformers library is not installed.")
    logging.error("Please install it with: pip install transformers torch")
    sys.exit(1)

def main():
    # Define paths
    script_dir = Path(__file__).parent
    judge_models_dir = script_dir / "evaluation" / "judgeblender"
    phi3_model_dir = judge_models_dir / "Phi-3-mini-4k-instruct"
    
    # Create directory if it doesn't exist
    if not judge_models_dir.exists():
        logging.warning(f"Creating judgeblender directory: {judge_models_dir}")
        os.makedirs(judge_models_dir, exist_ok=True)
    
    # Create model directory
    os.makedirs(phi3_model_dir, exist_ok=True)
    
    # Check if model already exists
    if list(phi3_model_dir.glob("*.bin")) or list(phi3_model_dir.glob("*.safetensors")):
        logging.warning("Model files already exist in the target directory.")
        user_input = input("Do you want to re-download the model? (y/n): ")
        if user_input.lower() != 'y':
            logging.info("Aborting download.")
            return
    
    logging.info(f"Downloading Phi-3-mini model to {phi3_model_dir}...")
    
    # Download and save the model and tokenizer
    model_id = "microsoft/Phi-3-mini-4k-instruct"
    
    try:
        # Download tokenizer
        logging.info("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # Download model with progress bar
        logging.info("Downloading model (this may take a while)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            device_map=None  # Don't load to GPU during download
        )
        
        logging.info("Saving tokenizer locally...")
        tokenizer.save_pretrained(phi3_model_dir)
        
        logging.info("Saving model locally (this may take a while)...")
        model.save_pretrained(phi3_model_dir)
        
        logging.info(f"Model and tokenizer saved successfully to {phi3_model_dir}")
        logging.info("You can now use the Phi-3 model in the Judge Blender system.")
    except Exception as e:
        logging.error(f"Error downloading model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Judge Blender Evaluation System

This module implements a blended judgment system using two LLMs to evaluate search results
from different search engine variants. The judges evaluate the relevance of search results
on a scale of 1-4.
"""

import os
import sys
import json
import time
import random
import argparse
from pathlib import Path
import logging
from typing import List, Dict, Any, Optional, Tuple
import torch
from torch import bfloat16
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import re

# Add project root to path to allow imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Constants
RESULTS_DIR = Path(__file__).parent / "results"
EVALUATION_DIR = Path(__file__).parent / "evaluations"
JUDGE_MODELS_DIR = Path(__file__).parent / "judgeblender"
TEMP_DIR = Path(__file__).parent / "temp"
DEFAULT_BATCH_SIZE = 8  # Number of prompts to process in one function call
MODEL_BATCH_SIZE = 8      # Number of prompts to process in parallel on GPU
DEFAULT_RESULTS_FILE = RESULTS_DIR / "pooled_results.json"

# Create temp directory if it doesn't exist
TEMP_DIR.mkdir(exist_ok=True)

# Relevance scale definition
RELEVANCE_SCALE = {
    1: "Not relevant - The document does not contain information related to the query",
    2: "Marginally relevant - The document mentions query terms but lacks substantial information",
    3: "Relevant - The document contains information related to the query but may not be comprehensive",
    4: "Highly relevant - The document directly addresses the query topic with comprehensive information"
}

class JudgeModel:
    """Base class for judge LLM models."""
    
    def __init__(self, model_path: str):
        """
        Initialize the judge model.
        
        Args:
            model_path: Path to the model directory
        """
        self.model_path = model_path
        self.model_name = Path(model_path).name
        self.pipe = None
        
    def load_model(self):
        """Load the model and tokenizer."""
        raise NotImplementedError("Subclasses must implement this method")
        
    def evaluate(self, prompt: str) -> Dict[str, Any]:
        """
        Evaluate a search result.
        
        Args:
            prompt: Formatted prompt for evaluation
            
        Returns:
            Evaluation result
        """
        raise NotImplementedError("Subclasses must implement this method")
        
    def evaluate_batch(self, prompts: List[str], batch_size: int = DEFAULT_BATCH_SIZE) -> List[Dict[str, Any]]:
        """
        Evaluate a batch of search results.
        
        Args:
            prompts: List of formatted prompts for evaluation
            batch_size: Size of batches to process
            
        Returns:
            List of evaluation results
        """
        raise NotImplementedError("Subclasses must implement this method")
        
    def extract_score(self, response: str) -> int:
        """
        Extract relevance scores from the model's response.
        
        Args:
            response: Model's text response
            
        Returns:
            Dictionary with dimension scores and overall score, or single overall score for backward compatibility
        """
        # Look for the structured format with dimension scores
        dimension_scores = {}
        dimension_names = ["keyword relevance", "search intent alignment", "expected utility", "overall relevance"]
        
        # Find all dimension scores in the response
        for dimension in dimension_names:
            pattern = rf"{dimension}:?\s*(\d)"
            match = re.search(pattern, response.lower())
            if match:
                try:
                    score = int(match.group(1))
                    if 1 <= score <= 4:
                        dimension_scores[dimension] = score
                except (ValueError, IndexError):
                    continue
        
        # If we have an overall score, return it
        if "overall relevance" in dimension_scores:
            return dimension_scores["overall relevance"]
        
        # If we have at least one dimension score but no overall, calculate a weighted average
        if dimension_scores:
            # Use weights that emphasize query alignment and keyword relevance
            weights = {
                "keyword relevance": 0.3,
                "search intent alignment": 0.4,
                "expected utility": 0.3
            }
            
            total_weight = 0
            weighted_sum = 0
            
            for dim, score in dimension_scores.items():
                if dim in weights:
                    weighted_sum += score * weights[dim]
                    total_weight += weights[dim]
            
            if total_weight > 0:
                # Round to nearest integer
                return round(weighted_sum / total_weight)
        
        # Fall back to the old approach if no structured scores found
        for line in response.split('\n'):
            line = line.strip()
            if line.startswith("Score:") or line.startswith("Relevance Score:"):
                try:
                    score_text = line.split(':')[1].strip()
                    score = int(score_text[0])  # Take first digit
                    if 1 <= score <= 4:
                        return score
                except (ValueError, IndexError):
                    continue
                    
        # If no clear score, look for keywords
        lower_resp = response.lower()
        if "excellent" in lower_resp or "highly relevant" in lower_resp:
            return 4
        elif "good" in lower_resp or "relevant" in lower_resp:
            return 3
        elif "fair" in lower_resp or "marginally" in lower_resp or "somewhat" in lower_resp:
            return 2
        else:
            return 1  # Default to not relevant
            
    def unload_model(self):
        """Unload the model and free GPU memory."""
        if self.pipe is not None:
            del self.pipe.model
            del self.pipe
            self.pipe = None
            torch.cuda.empty_cache()
            logging.info(f"Unloaded {self.model_name} model and cleared CUDA cache")


class GemmaJudge(JudgeModel):
    """Judge model using Gemma 7B Instruct."""
    
    def load_model(self):
        """Load the Gemma model and tokenizer."""
        model_path = os.path.join(JUDGE_MODELS_DIR, "gemma-7b-it")
        logging.info(f"Loading Gemma model from {model_path}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=bfloat16,
            device_map={"": 0},  # Explicitly map to CUDA:0
        )
        
        self.pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=32,
            do_sample=False,
            batch_size=MODEL_BATCH_SIZE  # Enable proper GPU batch processing
        )
        logging.info("Gemma model loaded successfully")
        
    def evaluate(self, prompt: str) -> Dict[str, Any]:
        """Evaluate using Gemma model."""
        full_prompt = f"""<start_of_turn>user
{prompt}
<end_of_turn>
<start_of_turn>model
"""
        
        start_time = time.time()
        response = self.pipe(full_prompt)[0]['generated_text']
        # Extract only the model's response
        response = response.split("<start_of_turn>model")[1].strip()
        end_time = time.time()
        
        score = self.extract_score(response)
        
        return {
            "model": "gemma-7b-it",
            "score": score,
            "response": response,
            "processing_time": end_time - start_time
        }
        
    def evaluate_batch(self, prompts: List[str], batch_size: int = DEFAULT_BATCH_SIZE) -> List[Dict[str, Any]]:
        """Evaluate a batch of prompts using Gemma model."""
        results = []
        
        # Process prompts in batches
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            formatted_prompts = [
                f"""<start_of_turn>user
{prompt}
<end_of_turn>
<start_of_turn>model
"""
                for prompt in batch_prompts
            ]
            
            start_time = time.time()
            # Debug the structure of batch_responses
            try:
                batch_outputs = self.pipe(formatted_prompts)
                # Clear CUDA cache after batch processing to free GPU memory
                torch.cuda.empty_cache()
                end_time = time.time()
                total_time = end_time - start_time
                
                # Process each response in the batch
                for j, output in enumerate(batch_outputs):
                    try:
                        # Handle different output formats - could be a dict or a list with dict
                        if isinstance(output, list) and len(output) > 0:
                            response_text = output[0]['generated_text']
                        else:
                            response_text = output['generated_text']
                            
                        # Extract only the model's response
                        response = response_text.split("<start_of_turn>model")[1].strip()
                        
                        score = self.extract_score(response)
                        
                        # Individual processing time is estimated
                        per_item_time = total_time / len(batch_prompts)
                        
                        results.append({
                            "model": "gemma-7b-it",
                            "score": score,
                            "response": response,
                            "processing_time": per_item_time
                        })
                    except (IndexError, TypeError, KeyError) as e:
                        error_msg = f"Error processing individual response {j} in batch {i//batch_size}: {e}"
                        logging.error(error_msg)
                        if hasattr(output, 'keys'):
                            logging.error(f"Output keys: {list(output.keys())}")
                        else:
                            logging.error(f"Output type: {type(output)}")
                        
                        # Add a failed evaluation
                        results.append({
                            "model": "gemma-7b-it",
                            "score": 0,
                            "response": f"Error: {error_msg}",
                            "processing_time": 0
                        })
            except Exception as e:
                error_msg = f"Fatal error processing batch {i//batch_size}: {e}"
                logging.error(error_msg)
                
                # Add failed evaluations for the entire batch
                for _ in range(len(batch_prompts)):
                    results.append({
                        "model": "gemma-7b-it",
                        "score": 0,
                        "response": f"Error: {error_msg}",
                        "processing_time": 0
                    })
            
            # Clear CUDA cache again at the end of the batch loop
            torch.cuda.empty_cache()
        
        return results


class MistralJudge(JudgeModel):
    """Judge model using Mistral 7B Instruct."""
    
    def load_model(self):
        """Load the Mistral model and tokenizer."""
        model_path = os.path.join(JUDGE_MODELS_DIR, "Mistral-7B-Instruct-v0.3")
        logging.info(f"Loading Mistral model from {model_path}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=bfloat16,
            device_map={"": 0},  # Explicitly map to CUDA:0
        )
        
        # Set pad_token_id to eos_token_id to fix batching issues
        if tokenizer.pad_token is None:
            logging.info("Setting Mistral tokenizer pad_token_id to eos_token_id")
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        self.pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=32,
            do_sample=False,
            batch_size=MODEL_BATCH_SIZE  # Enable proper GPU batch processing
        )
        logging.info("Mistral model loaded successfully")
        
    def evaluate(self, prompt: str) -> Dict[str, Any]:
        """Evaluate using Mistral model."""
        full_prompt = f"""<s>[INST] {prompt} [/INST]"""
        
        start_time = time.time()
        response = self.pipe(full_prompt)[0]['generated_text']
        # Extract only the model's response
        response = response.split('[/INST]')[1].strip()
        end_time = time.time()
        
        score = self.extract_score(response)
        
        return {
            "model": "Mistral-7B-Instruct-v0.3",
            "score": score,
            "response": response,
            "processing_time": end_time - start_time
        }
        
    def evaluate_batch(self, prompts: List[str], batch_size: int = DEFAULT_BATCH_SIZE) -> List[Dict[str, Any]]:
        """Evaluate a batch of prompts using Mistral model."""
        results = []
        
        # Process prompts in batches
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            formatted_prompts = [
                f"""<s>[INST] {prompt} [/INST]"""
                for prompt in batch_prompts
            ]
            
            start_time = time.time()
            # Debug the structure of batch_responses
            try:
                batch_outputs = self.pipe(formatted_prompts)
                # Clear CUDA cache after batch processing to free GPU memory
                torch.cuda.empty_cache()
                end_time = time.time()
                total_time = end_time - start_time
                
                # Process each response in the batch
                for j, output in enumerate(batch_outputs):
                    try:
                        # Handle different output formats - could be a dict or a list with dict
                        if isinstance(output, list) and len(output) > 0:
                            response_text = output[0]['generated_text']
                        else:
                            response_text = output['generated_text']
                            
                        # Extract only the model's response
                        response = response_text.split('[/INST]')[1].strip()
                        
                        score = self.extract_score(response)
                        
                        # Individual processing time is estimated
                        per_item_time = total_time / len(batch_prompts)
                        
                        results.append({
                            "model": "Mistral-7B-Instruct-v0.3",
                            "score": score,
                            "response": response,
                            "processing_time": per_item_time
                        })
                    except (IndexError, TypeError, KeyError) as e:
                        error_msg = f"Error processing individual response {j} in batch {i//batch_size}: {e}"
                        logging.error(error_msg)
                        if hasattr(output, 'keys'):
                            logging.error(f"Output keys: {list(output.keys())}")
                        else:
                            logging.error(f"Output type: {type(output)}")
                        
                        # Add a failed evaluation
                        results.append({
                            "model": "Mistral-7B-Instruct-v0.3",
                            "score": 0,
                            "response": f"Error: {error_msg}",
                            "processing_time": 0
                        })
            except Exception as e:
                error_msg = f"Fatal error processing batch {i//batch_size}: {e}"
                logging.error(error_msg)
                
                # Add failed evaluations for the entire batch
                for _ in range(len(batch_prompts)):
                    results.append({
                        "model": "Mistral-7B-Instruct-v0.3",
                        "score": 0,
                        "response": f"Error: {error_msg}",
                        "processing_time": 0
                    })
            
            # Clear CUDA cache again at the end of the batch loop
            torch.cuda.empty_cache()
        
        return results


class Phi3Judge(JudgeModel):
    """Judge model using Microsoft Phi-3-mini."""
    
    def load_model(self):
        """Load the Phi-3 model and tokenizer."""
        model_path = os.path.join(JUDGE_MODELS_DIR, "Phi-3-mini-4k-instruct")
        logging.info(f"Loading Phi-3 model from {model_path}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=bfloat16,
            device_map={"": 0},  # Explicitly map to CUDA:0
        )
        
        self.pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=32,
            do_sample=False,
            batch_size=MODEL_BATCH_SIZE  # Enable proper GPU batch processing
        )
        logging.info("Phi-3 model loaded successfully")
        
    def evaluate(self, prompt: str) -> Dict[str, Any]:
        """Evaluate using Phi-3 model."""
        full_prompt = f"""<|user|>
{prompt}
<|assistant|>"""
        
        start_time = time.time()
        response = self.pipe(full_prompt)[0]['generated_text']
        # Extract only the model's response
        response = response.split("<|assistant|>")[1].strip()
        end_time = time.time()
        
        score = self.extract_score(response)
        
        return {
            "model": "Phi-3-mini",
            "score": score,
            "response": response,
            "processing_time": end_time - start_time
        }
        
    def evaluate_batch(self, prompts: List[str], batch_size: int = DEFAULT_BATCH_SIZE) -> List[Dict[str, Any]]:
        """Evaluate a batch of prompts using Phi-3 model."""
        results = []
        
        # Process prompts in batches
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            formatted_prompts = [
                f"""<|user|>
{prompt}
<|assistant|>"""
                for prompt in batch_prompts
            ]
            
            start_time = time.time()
            try:
                batch_outputs = self.pipe(formatted_prompts)
                # Clear CUDA cache after batch processing to free GPU memory
                torch.cuda.empty_cache()
                end_time = time.time()
                total_time = end_time - start_time
                
                # Process each response in the batch
                for j, output in enumerate(batch_outputs):
                    try:
                        # Handle different output formats - could be a dict or a list with dict
                        if isinstance(output, list) and len(output) > 0:
                            response_text = output[0]['generated_text']
                        else:
                            response_text = output['generated_text']
                            
                        # Extract only the model's response
                        response = response_text.split("<|assistant|>")[1].strip()
                        
                        score = self.extract_score(response)
                        
                        # Individual processing time is estimated
                        per_item_time = total_time / len(batch_prompts)
                        
                        results.append({
                            "model": "Phi-3-mini",
                            "score": score,
                            "response": response,
                            "processing_time": per_item_time
                        })
                    except (IndexError, TypeError, KeyError) as e:
                        error_msg = f"Error processing individual response {j} in batch {i//batch_size}: {e}"
                        logging.error(error_msg)
                        if hasattr(output, 'keys'):
                            logging.error(f"Output keys: {list(output.keys())}")
                        else:
                            logging.error(f"Output type: {type(output)}")
                        
                        # Add a failed evaluation
                        results.append({
                            "model": "Phi-3-mini",
                            "score": 0,
                            "response": f"Error: {error_msg}",
                            "processing_time": 0
                        })
            except Exception as e:
                error_msg = f"Fatal error processing batch {i//batch_size}: {e}"
                logging.error(error_msg)
                
                # Add failed evaluations for the entire batch
                for _ in range(len(batch_prompts)):
                    results.append({
                        "model": "Phi-3-mini",
                        "score": 0,
                        "response": f"Error: {error_msg}",
                        "processing_time": 0
                    })
            
            # Clear CUDA cache again at the end of the batch loop
            torch.cuda.empty_cache()
        
        return results


def format_evaluation_prompt(query_data: Dict[str, Any], result: Dict[str, Any]) -> str:
    """
    Format a prompt for evaluating a search result.
    
    Args:
        query_data: Test query data
        result: Search result to evaluate
        
    Returns:
        Formatted prompt
    """
    query_text = query_data['query']
    topic_name = query_data['topic_name']
    topic_keywords = query_data['topic_keywords']
    topic_representation = query_data.get('topic_representation', 'N/A')
    representation_category = query_data.get('representation_category', 'unknown')
    is_interdisciplinary = query_data.get('is_interdisciplinary', False)
    
    title = result.get('title', 'No title available')
    abstract = result.get('abstract', '')  # May be empty
    paper_id = result.get('paper_id', 'Unknown')
    rank = result.get('rank', 'Unknown')
    score = result.get('score', 'Unknown')
    year = result.get('year', '')  # May be empty
    authors = result.get('authors', [])
    author_text = ", ".join(authors) if authors else ""  # May be empty
    
    # Build available metadata text conditionally
    metadata_parts = []
    if abstract:
        metadata_parts.append(f"Abstract: {abstract}")
    if author_text:
        metadata_parts.append(f"Authors: {author_text}")
    if year:
        metadata_parts.append(f"Year: {year}")
    
    metadata_text = "\n".join(metadata_parts)
    if metadata_text:
        metadata_text = f"\n{metadata_text}"
    
    # Format prompt for multi-dimensional evaluation with 3 dimensions
    prompt = f"""You are a judge evaluating the relevance of an academic search result for a query.

QUERY: "{query_text}"

SEARCH RESULT:
Title: {title}{metadata_text}
Original Rank: {rank}
Engine Score: {score}

ADDITIONAL CONTEXT:
Topic: {topic_name}
Topic Keywords: {', '.join(topic_keywords)}
Topic Representation: {topic_representation} ({representation_category} representation category)
Interdisciplinary Topic: {'Yes' if is_interdisciplinary else 'No'}

Rate this search result on three dimensions using scores of 1-4 (1=Poor, 2=Fair, 3=Good, 4=Excellent):

1. KEYWORD RELEVANCE: How well the title/abstract match the query keywords
2. SEARCH INTENT ALIGNMENT: How well the document addresses the query's information need
3. EXPECTED UTILITY: How useful this document would be to a researcher

Provide ratings WITHOUT explanations in this exact format:
Keyword Relevance: [SCORE]
Search Intent Alignment: [SCORE]
Expected Utility: [SCORE]
Overall Relevance: [SCORE]
"""
    return prompt


def run_sequential_evaluation(results_file: Path, batch_size: int = DEFAULT_BATCH_SIZE) -> Dict[str, Any]:
    """
    Run evaluation sequentially, one model at a time, with batch processing.
    
    Args:
        results_file: Path to the results file to evaluate
        batch_size: Number of prompts to process in each batch
        
    Returns:
        Evaluation results
    """
    # Load results from file
    with open(results_file, 'r') as f:
        results_data = json.load(f)
    
    # Extract test queries and search results
    test_queries = results_data.get('test_queries', {})
    search_results = results_data.get('search_results', {})
    
    # Set up data structure for evaluations
    evaluations = {
        'file': results_file.name,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'system_name': results_data.get('system_name', 'unknown'),
        'query_evaluations': {}
    }
    
    # Create list of all evaluation tasks
    eval_tasks = []
    for query_id, query_data in test_queries.items():
        if query_id not in search_results:
            logging.warning(f"Query {query_id} not found in search results")
            continue
        
        engine_results = search_results[query_id]
        for result_index, result in enumerate(engine_results):
            eval_tasks.append((query_id, query_data, result, result_index))
    
    total_tasks = len(eval_tasks)
    logging.info(f"Prepared {total_tasks} search result evaluation tasks")
    logging.info(f"Using batch size: {batch_size}")
    
    # Dictionary to store results for each model
    model_results = {}
    
    # Sequential evaluation: Process Gemma first
    gemma_judge = GemmaJudge(os.path.join(JUDGE_MODELS_DIR, "gemma-7b-it"))
    gemma_judge.load_model()
    
    # Process all evaluation tasks with Gemma
    try:
        logging.info("Starting evaluations with Gemma model...")
        gemma_results = {}
        
        # Prepare all prompts in advance
        all_prompts = []
        task_indices = []  # Keep track of which prompt corresponds to which task
        
        for i, (query_id, query_data, result, result_index) in enumerate(eval_tasks):
            prompt = format_evaluation_prompt(query_data, result)
            all_prompts.append(prompt)
            task_indices.append((query_id, result_index))
            
        # Process in batches
        logging.info(f"Processing {len(all_prompts)} Gemma evaluations in batches of {batch_size}")
        processed = 0
        
        for i in range(0, len(all_prompts), batch_size):
            batch_prompts = all_prompts[i:i+batch_size]
            batch_indices = task_indices[i:i+batch_size]
            
            batch_num = i//batch_size + 1
            total_batches = (len(all_prompts) + batch_size - 1)//batch_size
            logging.info(f"Starting Gemma batch {batch_num}/{total_batches} at {time.strftime('%H:%M:%S')}")
            batch_start_time = time.time()
            
            try:
                batch_results = gemma_judge.evaluate_batch(batch_prompts, batch_size)
                batch_end_time = time.time()
                batch_duration = batch_end_time - batch_start_time
                
                for j, evaluation in enumerate(batch_results):
                    query_id, result_index = batch_indices[j]
                    result_id = f"{query_id}_{result_index}"
                    gemma_results[result_id] = evaluation
                
                # Update progress
                processed += len(batch_prompts)
                items_per_second = len(batch_prompts) / batch_duration if batch_duration > 0 else 0
                remaining_batches = total_batches - batch_num
                est_remaining_time = (remaining_batches * batch_duration) / 60  # minutes
                
                logging.info(f"Gemma: Batch {batch_num} completed in {batch_duration:.2f}s ({items_per_second:.2f} items/sec)")
                logging.info(f"Gemma: Processed {processed}/{total_tasks} ({processed/total_tasks*100:.1f}%)")
                logging.info(f"Gemma: Est. remaining time: {est_remaining_time:.1f} minutes")
                
            except Exception as e:
                logging.error(f"Error evaluating batch {batch_num}/{total_batches} with Gemma: {e}")
        
        # Save Gemma results to temp file
        with open(TEMP_DIR / "gemma_results.json", 'w') as f:
            json.dump(gemma_results, f)
            
        logging.info("Completed Gemma evaluations")
        model_results["gemma"] = gemma_results
    finally:
        # Unload Gemma model to free memory
        gemma_judge.unload_model()
    
    # Process with Mistral next
    mistral_judge = MistralJudge(os.path.join(JUDGE_MODELS_DIR, "Mistral-7B-Instruct-v0.3"))
    mistral_judge.load_model()
    
    # Process all evaluation tasks with Mistral
    try:
        logging.info("Starting evaluations with Mistral model...")
        mistral_results = {}
        
        # Prepare all prompts in advance (reuse the ones prepared for Gemma)
        # Process in batches
        logging.info(f"Processing {len(all_prompts)} Mistral evaluations in batches of {batch_size}")
        processed = 0
        
        for i in range(0, len(all_prompts), batch_size):
            batch_prompts = all_prompts[i:i+batch_size]
            batch_indices = task_indices[i:i+batch_size]
            
            batch_num = i//batch_size + 1
            total_batches = (len(all_prompts) + batch_size - 1)//batch_size
            logging.info(f"Starting Mistral batch {batch_num}/{total_batches} at {time.strftime('%H:%M:%S')}")
            batch_start_time = time.time()
            
            try:
                batch_results = mistral_judge.evaluate_batch(batch_prompts, batch_size)
                batch_end_time = time.time()
                batch_duration = batch_end_time - batch_start_time
                
                for j, evaluation in enumerate(batch_results):
                    query_id, result_index = batch_indices[j]
                    result_id = f"{query_id}_{result_index}"
                    mistral_results[result_id] = evaluation
                
                # Update progress
                processed += len(batch_prompts)
                items_per_second = len(batch_prompts) / batch_duration if batch_duration > 0 else 0
                remaining_batches = total_batches - batch_num
                est_remaining_time = (remaining_batches * batch_duration) / 60  # minutes
                
                logging.info(f"Mistral: Batch {batch_num} completed in {batch_duration:.2f}s ({items_per_second:.2f} items/sec)")
                logging.info(f"Mistral: Processed {processed}/{total_tasks} ({processed/total_tasks*100:.1f}%)")
                logging.info(f"Mistral: Est. remaining time: {est_remaining_time:.1f} minutes")
                
            except Exception as e:
                logging.error(f"Error evaluating batch {batch_num}/{total_batches} with Mistral: {e}")
        
        # Save Mistral results to temp file
        with open(TEMP_DIR / "mistral_results.json", 'w') as f:
            json.dump(mistral_results, f)
            
        logging.info("Completed Mistral evaluations")
        model_results["mistral"] = mistral_results
    finally:
        # Unload Mistral model to free memory
        mistral_judge.unload_model()
    
    # Process with Phi-3 next
    phi3_judge = Phi3Judge(os.path.join(JUDGE_MODELS_DIR, "Phi-3-mini-4k-instruct"))
    phi3_judge.load_model()
    
    # Process all evaluation tasks with Phi-3
    try:
        logging.info("Starting evaluations with Phi-3 model...")
        phi3_results = {}
        
        # Process in batches
        logging.info(f"Processing {len(all_prompts)} Phi-3 evaluations in batches of {batch_size}")
        processed = 0
        
        for i in range(0, len(all_prompts), batch_size):
            batch_prompts = all_prompts[i:i+batch_size]
            batch_indices = task_indices[i:i+batch_size]
            
            batch_num = i//batch_size + 1
            total_batches = (len(all_prompts) + batch_size - 1)//batch_size
            logging.info(f"Starting Phi-3 batch {batch_num}/{total_batches} at {time.strftime('%H:%M:%S')}")
            batch_start_time = time.time()
            
            try:
                batch_results = phi3_judge.evaluate_batch(batch_prompts, batch_size)
                batch_end_time = time.time()
                batch_duration = batch_end_time - batch_start_time
                
                for j, evaluation in enumerate(batch_results):
                    query_id, result_index = batch_indices[j]
                    result_id = f"{query_id}_{result_index}"
                    phi3_results[result_id] = evaluation
                
                # Update progress
                processed += len(batch_prompts)
                items_per_second = len(batch_prompts) / batch_duration if batch_duration > 0 else 0
                remaining_batches = total_batches - batch_num
                est_remaining_time = (remaining_batches * batch_duration) / 60  # minutes
                
                logging.info(f"Phi-3: Batch {batch_num} completed in {batch_duration:.2f}s ({items_per_second:.2f} items/sec)")
                logging.info(f"Phi-3: Processed {processed}/{total_tasks} ({processed/total_tasks*100:.1f}%)")
                logging.info(f"Phi-3: Est. remaining time: {est_remaining_time:.1f} minutes")
                
            except Exception as e:
                logging.error(f"Error evaluating batch {batch_num}/{total_batches} with Phi-3: {e}")
        
        # Save Phi-3 results to temp file
        with open(TEMP_DIR / "phi3_results.json", 'w') as f:
            json.dump(phi3_results, f)
            
        logging.info("Completed Phi-3 evaluations")
        model_results["phi3"] = phi3_results
    finally:
        # Unload Phi-3 model to free memory
        phi3_judge.unload_model()
    
    # Combine results from both models
    for query_id, query_data in test_queries.items():
        if query_id not in search_results:
            continue
        
        engine_results = search_results[query_id]
        evaluations['query_evaluations'][query_id] = {
            'query': query_data['query'],
            'topic_name': query_data.get('topic_name', 'unknown'),
            'result_evaluations': []
        }
        
        for result_index, result in enumerate(engine_results):
            result_id = f"{query_id}_{result_index}"
            
            # Get evaluations from both models
            gemma_eval = model_results.get("gemma", {}).get(result_id, {})
            mistral_eval = model_results.get("mistral", {}).get(result_id, {})
            phi3_eval = model_results.get("phi3", {}).get(result_id, {})
            
            # Calculate blended score
            gemma_score = gemma_eval.get('score', 0)
            mistral_score = mistral_eval.get('score', 0)
            phi3_score = phi3_eval.get('score', 0)
            
            # If models failed, use the available scores; otherwise take average
            valid_scores = []
            if gemma_score > 0:
                valid_scores.append(gemma_score)
            if mistral_score > 0:
                valid_scores.append(mistral_score)
            if phi3_score > 0:
                valid_scores.append(phi3_score)
            
            blended_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0
            
            evaluation_entry = {
                'result_index': result_index,
                'title': result.get('title', 'No title'),
                'gemma_evaluation': gemma_eval,
                'mistral_evaluation': mistral_eval,
                'phi3_evaluation': phi3_eval,
                'blended_score': blended_score
            }
            
            evaluations['query_evaluations'][query_id]['result_evaluations'].append(evaluation_entry)
    
    logging.info("Completed blending of evaluations")
    return evaluations


def create_evaluation_summary(evaluations: Dict[str, Any], output_file: Path) -> None:
    """
    Create a summary of the evaluations.
    
    Args:
        evaluations: Evaluation results
        output_file: Path to save the summary
    """
    summary = {
        'file': evaluations['file'],
        'timestamp': evaluations['timestamp'],
        'system_name': evaluations['system_name'],
        'query_summaries': {},
        'dimension_stats': {
            'keyword_relevance': {'sum': 0, 'count': 0},
            'search_intent_alignment': {'sum': 0, 'count': 0},
            'expected_utility': {'sum': 0, 'count': 0}
        }
    }
    
    # Initialize counters
    total_results = 0
    score_sum = 0
    score_distributions = {1: 0, 2: 0, 3: 0, 4: 0}
    
    # Process each query
    for query_id, query_eval in evaluations['query_evaluations'].items():
        query_results = query_eval['result_evaluations']
        query_total = len(query_results)
        
        # Skip if no results
        if query_total == 0:
            continue
        
        query_sum = sum(r['blended_score'] for r in query_results)
        query_avg = query_sum / query_total if query_total > 0 else 0
        
        # Count scores by range
        query_distribution = {1: 0, 2: 0, 3: 0, 4: 0}
        
        # Initialize dimension sums for this query
        query_dimension_stats = {
            'keyword_relevance': {'sum': 0, 'count': 0},
            'search_intent_alignment': {'sum': 0, 'count': 0},
            'expected_utility': {'sum': 0, 'count': 0}
        }
        
        for result in query_results:
            score = result['blended_score']
            rounded_score = round(score)
            if 1 <= rounded_score <= 4:
                query_distribution[rounded_score] += 1
                score_distributions[rounded_score] += 1
            
            # Extract dimension scores if available from Gemma and Mistral evaluations
            for model_name in ['gemma_evaluation', 'mistral_evaluation', 'phi3_evaluation']:
                if model_name in result:
                    model_eval = result[model_name]
                    response = model_eval.get('response', '')
                    
                    # Check for dimension scores in the response
                    dimensions = [
                        ('keyword_relevance', r"keyword relevance:?\s*(\d)"),
                        ('search_intent_alignment', r"search intent alignment:?\s*(\d)"),
                        ('expected_utility', r"expected utility:?\s*(\d)")
                    ]
                    
                    for dim_name, pattern in dimensions:
                        match = re.search(pattern, response.lower())
                        if match:
                            try:
                                dim_score = int(match.group(1))
                                if 1 <= dim_score <= 4:
                                    # Add to query stats
                                    query_dimension_stats[dim_name]['sum'] += dim_score
                                    query_dimension_stats[dim_name]['count'] += 1
                                    
                                    # Add to overall stats
                                    summary['dimension_stats'][dim_name]['sum'] += dim_score
                                    summary['dimension_stats'][dim_name]['count'] += 1
                            except (ValueError, IndexError):
                                continue
        
        # Calculate dimension averages for this query
        query_dimension_averages = {}
        for dim_name, stats in query_dimension_stats.items():
            if stats['count'] > 0:
                query_dimension_averages[dim_name] = stats['sum'] / stats['count']
        
        # Update summary for this query
        summary['query_summaries'][query_id] = {
            'query': query_eval['query'],
            'topic_name': query_eval['topic_name'],
            'result_count': query_total,
            'average_score': query_avg,
            'score_distribution': query_distribution,
            'dimension_averages': query_dimension_averages
        }
        
        # Update totals
        total_results += query_total
        score_sum += query_sum
    
    # Calculate overall average
    overall_avg = score_sum / total_results if total_results > 0 else 0
    
    # Calculate dimension averages across all queries
    dimension_averages = {}
    for dim_name, stats in summary['dimension_stats'].items():
        if stats['count'] > 0:
            dimension_averages[dim_name] = stats['sum'] / stats['count']
    
    # Add overall summary
    summary['overall'] = {
        'total_queries': len(summary['query_summaries']),
        'total_results': total_results,
        'average_score': overall_avg,
        'score_distribution': score_distributions,
        'dimension_averages': dimension_averages
    }
    
    # Save summary
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logging.info(f"Evaluation summary saved to {output_file}")


def main():
    """Main entry point for the judge blender."""
    parser = argparse.ArgumentParser(description="Judge Blender Evaluation System")
    parser.add_argument(
        "--results-file",
        type=str,
        default=str(DEFAULT_RESULTS_FILE),
        help="Path to JSON file containing search results to evaluate (default: ./results/pooled_results.json)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save evaluation results (default: auto-generated in evaluations dir)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Number of prompts to process in each batch (default: {DEFAULT_BATCH_SIZE})"
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    results_path = Path(args.results_file)
    if not results_path.exists():
        logging.error(f"Results file not found: {results_path}")
        sys.exit(1)
    
    # Auto-generate output path if not specified
    if args.output:
        output_path = Path(args.output)
    else:
        # Create output filename based on input with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = EVALUATION_DIR / f"{results_path.stem}_eval_{timestamp}.json"
    
    # Create parent directories if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Run sequential evaluation
    logging.info(f"Starting sequential evaluation of {results_path}")
    evaluations = run_sequential_evaluation(results_path, batch_size=args.batch_size)
    
    # Save full evaluation results
    with open(output_path, 'w') as f:
        json.dump(evaluations, f, indent=2)
    
    logging.info(f"Evaluations saved to {output_path}")
    
    # Generate and save summary
    summary_path = output_path.parent / f"{output_path.stem}_summary.json"
    create_evaluation_summary(evaluations, summary_path)
    
    logging.info("Evaluation complete")


if __name__ == "__main__":
    main() 
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
DEFAULT_BATCH_SIZE = 1
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
        Extract a relevance score from the model's response.
        
        Args:
            response: Model's text response
            
        Returns:
            Relevance score (1-4)
        """
        # Look for a number 1-4 in the response
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
        if "highly relevant" in lower_resp or "very relevant" in lower_resp:
            return 4
        elif "relevant" in lower_resp and "not relevant" not in lower_resp:
            return 3
        elif "marginally" in lower_resp or "somewhat" in lower_resp:
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
            do_sample=False
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
        
        self.pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=32,
            do_sample=False
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
    abstract = result.get('abstract', 'No abstract available')
    
    # Comment out full text inclusion to speed up processing
    """
    # Check for full text in various possible field names
    full_text = None
    for field in ['fullText', 'body_text', 'full_text', 'text']:
        if field in result and result[field] and result[field] is not None:
            full_text = result[field]
            break
    
    # Format full text for inclusion (truncate if too long)
    text_section = ""
    if full_text:
        # Truncate full text if it's very long (max 1000 chars)
        if isinstance(full_text, str) and full_text.strip():
            truncated_text = full_text[:1000] + "..." if len(full_text) > 1000 else full_text
            text_section = f"\nFull Text (excerpt):\n{truncated_text}"
    """
    
    # Set text_section to empty string since we're skipping full text
    text_section = ""
    
    # Format prompt
    prompt = f"""You are a judge evaluating the relevance of a search result for an academic search query.

QUERY: "{query_text}"

SEARCH RESULT:
Title: {title}
Abstract: {abstract}{text_section}

ADDITIONAL CONTEXT:
Topic: {topic_name}
Topic Keywords: {', '.join(topic_keywords)}
Topic Representation: {topic_representation} ({representation_category} representation category)
Interdisciplinary Topic: {'Yes' if is_interdisciplinary else 'No'}

Please evaluate the relevance of this search result to the query on a scale of 1-4:
1 = Not relevant - The document does not contain information related to the query
2 = Marginally relevant - The document mentions query terms but lacks substantial information
3 = Relevant - The document contains information related to the query but may not be comprehensive
4 = Highly relevant - The document directly addresses the query topic with comprehensive information

Provide your relevance score and a brief explanation of your reasoning. Format your response starting with "Relevance Score: X" followed by your explanation.
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
            
            # Get evaluations from both models if available
            gemma_eval = model_results.get("gemma", {}).get(result_id, {})
            mistral_eval = model_results.get("mistral", {}).get(result_id, {})
            
            # Calculate blended score
            gemma_score = gemma_eval.get('score', 0)
            mistral_score = mistral_eval.get('score', 0)
            
            # If one model failed, use the other's score; otherwise take average
            if gemma_score == 0:
                blended_score = mistral_score
            elif mistral_score == 0:
                blended_score = gemma_score
            else:
                blended_score = (gemma_score + mistral_score) / 2
            
            evaluation_entry = {
                'result_index': result_index,
                'title': result.get('title', 'No title'),
                'gemma_evaluation': gemma_eval,
                'mistral_evaluation': mistral_eval,
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
        'query_summaries': {}
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
        for result in query_results:
            score = result['blended_score']
            rounded_score = round(score)
            if 1 <= rounded_score <= 4:
                query_distribution[rounded_score] += 1
                score_distributions[rounded_score] += 1
            
        # Update summary for this query
        summary['query_summaries'][query_id] = {
            'query': query_eval['query'],
            'topic_name': query_eval['topic_name'],
            'result_count': query_total,
            'average_score': query_avg,
            'score_distribution': query_distribution
        }
        
        # Update totals
        total_results += query_total
        score_sum += query_sum
    
    # Calculate overall average
    overall_avg = score_sum / total_results if total_results > 0 else 0
    
    # Add overall summary
    summary['overall'] = {
        'total_queries': len(summary['query_summaries']),
        'total_results': total_results,
        'average_score': overall_avg,
        'score_distribution': score_distributions
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
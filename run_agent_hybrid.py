#!/usr/bin/env python3
"""
Main entry point for the hybrid retail analytics agent.

This script can run in two modes:
1. Single question mode: Use --question to ask one question interactively
2. Batch mode: Use --batch and --out to process a JSONL file of questions

The script automatically detects the project root and handles path resolution,
similar to the notebook implementation.
"""

import json
import logging
import sys
import click
from pathlib import Path
from typing import List, Dict, Any

from agent.graph_hybrid import HybridAgent


def setup_logging(verbose: bool = False):
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('agent.log')
        ]
    )


def load_questions(batch_file: str) -> List[Dict[str, Any]]:
    """Load questions from JSONL file."""
    questions = []
    with open(batch_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


def save_outputs(outputs: List[Dict[str, Any]], output_file: str):
    """Save outputs to JSONL file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        for output in outputs:
            f.write(json.dumps(output) + '\n')


def run_single_question(agent: 'HybridAgent', question: str, format_hint: str = "", question_id: str = "ad_hoc") -> Dict[str, Any]:
    """Helper function to run a single question with debugging output."""
    result = agent.process_question(question=question, format_hint=format_hint, question_id=question_id)
    
    # Print debug info if verbose logging is enabled
    logger = logging.getLogger(__name__)
    if logger.isEnabledFor(logging.DEBUG):
        final_answer = result.get('final_answer')
        logger.debug(f"ID: {result.get('id')}")
        logger.debug(f"Answer: {final_answer}")
        logger.debug(f"Confidence: {result.get('confidence')}")
        logger.debug(f"SQL rows: {len(result.get('sql', '')) > 0} | Citations: {result.get('citations')}")
    
    return result


def validate_output(output: Dict[str, Any], expected_format: str) -> bool:
    """Validate that output matches expected format."""
    final_answer = output.get("final_answer")
    
    if final_answer is None:
        return False
    
    try:
        if expected_format == "int":
            return isinstance(final_answer, int)
        elif expected_format == "float":
            return isinstance(final_answer, (int, float))
        elif expected_format.startswith("list[") or expected_format.startswith("List["):
            return isinstance(final_answer, list)
        elif expected_format.startswith("{") or expected_format == "dict":
            return isinstance(final_answer, dict)
        else:
            return True  # Unknown format, assume valid
    except:
        return False


@click.command()
@click.option('--batch', help='Path to JSONL file with questions')
@click.option('--out', help='Path to output JSONL file')
@click.option('--question', '-q', help='Single question to ask (alternative to batch mode)')
@click.option('--format-hint', default='', help='Format hint for single question')
@click.option('--db', default='data/northwind.sqlite', help='Path to SQLite database')
@click.option('--docs', default='docs/', help='Path to documents directory')
@click.option('--model', default='phi3.5:3.8b-mini-instruct-q4_K_M', help='Ollama model name')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--root', default=None, help='Project root directory (auto-detected if not provided)')
def main(batch: str, out: str, question: str, format_hint: str, db: str, docs: str, model: str, verbose: bool, root: str):
    """Run the hybrid retail analytics agent on a batch of questions or a single question."""
    
    # Validate arguments
    if not batch and not question:
        click.echo("Error: Must provide either --batch or --question", err=True)
        return
    
    if batch and question:
        click.echo("Error: Cannot use both --batch and --question at the same time", err=True)
        return
    
    if batch and not out:
        click.echo("Error: --out is required when using --batch", err=True)
        return
    
    # Set up logging
    setup_logging(verbose)
    
    logger = logging.getLogger(__name__)
    logger.info("Starting hybrid retail analytics agent")
    
    # Determine project root
    if root:
        project_root = Path(root)
    else:
        # Auto-detect project root (where this script is located)
        project_root = Path(__file__).parent.absolute()
    
    # Ensure project root is in Python path
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # Convert relative paths to absolute paths based on project root
    db_path = Path(db) if Path(db).is_absolute() else project_root / db
    docs_path = Path(docs) if Path(docs).is_absolute() else project_root / docs
    
    # Validate required paths
    if not db_path.exists():
        logger.error(f"Database file not found: {db_path}")
        return
    
    if not docs_path.exists():
        logger.error(f"Documents directory not found: {docs_path}")
        return
    
    # Initialize agent
    logger.info("Initializing hybrid agent")
    try:
        agent = HybridAgent(
            db_path=str(db_path),
            docs_dir=str(docs_path),
            model_name=model
        )
        logger.info("Agent initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}")
        return
    
    # Handle single question mode
    if question:
        logger.info(f"Processing single question: {question}")
        try:
            result = run_single_question(
                agent=agent,
                question=question,
                format_hint=format_hint,
                question_id="single_question"
            )
            
            # Print results
            print(f"\n{'='*60}")
            print(f"Question: {question}")
            print(f"{'='*60}")
            print(f"Answer: {result.get('final_answer')}")
            print(f"Confidence: {result.get('confidence', 0.0)}")
            if result.get('sql'):
                print(f"SQL Query: {result.get('sql')}")
            if result.get('explanation'):
                print(f"Explanation: {result.get('explanation')}")
            if result.get('citations'):
                print(f"Citations: {result.get('citations')}")
            print(f"{'='*60}")
            
        except Exception as e:
            logger.error(f"Failed to process question: {e}")
        
        return
    
    # Handle batch mode
    batch_path = Path(batch) if Path(batch).is_absolute() else project_root / batch
    out_path = Path(out) if Path(out).is_absolute() else project_root / out
    
    # Validate batch file
    if not batch_path.exists():
        logger.error(f"Batch file not found: {batch_path}")
        return
    
    # Load questions
    logger.info(f"Loading questions from {batch_path}")
    try:
        questions = load_questions(str(batch_path))
        logger.info(f"Loaded {len(questions)} questions")
    except Exception as e:
        logger.error(f"Failed to load questions: {e}")
        return
    
    # Process questions
    outputs = []
    successful = 0
    failed = 0
    
    for i, question_data in enumerate(questions, 1):
        question_id = question_data.get("id", f"question_{i}")
        question = question_data.get("question", "")
        format_hint = question_data.get("format_hint", "")
        
        logger.info(f"Processing question {i}/{len(questions)}: {question_id}")
        
        try:
            # Process the question using the helper function
            result = run_single_question(
                agent=agent,
                question=question,
                format_hint=format_hint,
                question_id=question_id
            )
            
            # Validate output format
            is_valid = validate_output(result, format_hint)
            
            if is_valid and result.get("final_answer") is not None:
                successful += 1
                logger.info(f"✓ Successfully processed {question_id}")
            else:
                failed += 1
                logger.warning(f"✗ Invalid output for {question_id}: {result.get('explanation', 'Unknown error')}")
            
            # Clean up result for output (remove trace_log to save space)
            output_result = {
                "id": result.get("id", question_id),
                "final_answer": result.get("final_answer"),
                "sql": result.get("sql", ""),
                "confidence": result.get("confidence", 0.0),
                "explanation": result.get("explanation", ""),
                "citations": result.get("citations", [])
            }
            
            outputs.append(output_result)
            
            # Log trace if verbose
            if verbose and result.get("trace_log"):
                logger.debug(f"Trace for {question_id}: {json.dumps(result['trace_log'], indent=2)}")
        
        except Exception as e:
            failed += 1
            logger.error(f"✗ Failed to process {question_id}: {e}")
            
            # Add error output
            outputs.append({
                "id": question_id,
                "final_answer": None,
                "sql": "",
                "confidence": 0.0,
                "explanation": f"Processing failed: {str(e)}",
                "citations": []
            })
    
    # Save outputs
    logger.info(f"Saving outputs to {out_path}")
    try:
        save_outputs(outputs, str(out_path))
        logger.info(f"Outputs saved successfully")
    except Exception as e:
        logger.error(f"Failed to save outputs: {e}")
        return
    
    # Summary
    logger.info(f"""
Processing Summary:
- Total questions: {len(questions)}
- Successful: {successful}
- Failed: {failed}
- Success rate: {successful/len(questions)*100:.1f}%
- Output file: {out_path}
    """)
    
    print(f"\n✓ Processed {len(questions)} questions")
    print(f"✓ Success rate: {successful}/{len(questions)} ({successful/len(questions)*100:.1f}%)")
    print(f"✓ Results saved to: {out_path}")


if __name__ == "__main__":
    main()

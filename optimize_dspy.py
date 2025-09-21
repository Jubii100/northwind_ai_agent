#!/usr/bin/env python3
"""DSPy optimization script for the hybrid agent."""

import json
import dspy
from typing import List, Dict, Any
from pathlib import Path

from agent.dspy_signatures import Router
from agent.graph_hybrid import HybridAgent


def create_training_data() -> List[Dict[str, Any]]:
    """Create training data for DSPy optimization."""
    
    # Sample training examples for router optimization
    training_examples = [
        # SQL-only questions
        {
            "question": "Top 5 customers by total revenue in 1997",
            "ontological_blocks": [{"content": "revenue calculation, customers, 1997", "tags": ["sql"]}],
            "expected_answer_type": "list",
            "expected_approach": "sql"
        },
        {
            "question": "Monthly revenue for 1997",
            "ontological_blocks": [{"content": "monthly aggregation, revenue, 1997", "tags": ["sql"]}],
            "expected_answer_type": "list",
            "expected_approach": "sql"
        },
        {
            "question": "Average discount percentage by product category",
            "ontological_blocks": [{"content": "discount analysis, categories", "tags": ["sql"]}],
            "expected_answer_type": "list",
            "expected_approach": "sql"
        },
        
        # RAG-only questions
        {
            "question": "What is the standard shipping window for domestic orders?",
            "ontological_blocks": [{"content": "shipping policy, domestic orders", "tags": ["rag"]}],
            "expected_answer_type": "int",
            "expected_approach": "rag"
        },
        {
            "question": "What is the return window for defective items?",
            "ontological_blocks": [{"content": "return policy, defective items", "tags": ["rag"]}],
            "expected_answer_type": "int",
            "expected_approach": "rag"
        },
        {
            "question": "What is the return window for unopened Beverages?",
            "ontological_blocks": [{"content": "return policy, beverages", "tags": ["rag"]}],
            "expected_answer_type": "int",
            "expected_approach": "rag"
        },
        
        # Hybrid questions
        {
            "question": "Total revenue from Beverages during Summer Beverages 1997 campaign",
            "ontological_blocks": [{"content": "summer campaign dates, beverages revenue", "tags": ["rag", "sql"]}],
            "expected_answer_type": "float",
            "expected_approach": "hybrid"
        },
        {
            "question": "AOV during Winter Classics 1997 campaign",
            "ontological_blocks": [{"content": "winter campaign dates, AOV definition", "tags": ["rag", "sql"]}],
            "expected_answer_type": "float",
            "expected_approach": "hybrid"
        },
        {
            "question": "Compare AOV for discounted vs non-discounted orders during Winter Classics 1997",
            "ontological_blocks": [{"content": "winter campaign, discount comparison, AOV", "tags": ["rag", "sql"]}],
            "expected_answer_type": "dict",
            "expected_approach": "hybrid"
        },
        {
            "question": "Estimated returns cost for Beverages during Summer Beverages 1997",
            "ontological_blocks": [{"content": "return policy, summer campaign, beverages", "tags": ["rag", "sql"]}],
            "expected_answer_type": "float",
            "expected_approach": "hybrid"
        }
    ]
    
    return training_examples


def evaluate_router(router: Router, test_data: List[Dict[str, Any]]) -> float:
    """Evaluate router accuracy on test data."""
    correct = 0
    total = len(test_data)
    
    for example in test_data:
        try:
            result = router.forward(
                question=example["question"],
                ontological_blocks=example["ontological_blocks"],
                expected_answer_type=example["expected_answer_type"]
            )
            
            if result.approach == example["expected_approach"]:
                correct += 1
        except Exception as e:
            print(f"Error evaluating example: {e}")
            continue
    
    return correct / total if total > 0 else 0.0


def optimize_router():
    """Optimize the router module using DSPy."""
    print("Starting Router optimization...")
    
    # Initialize DSPy with Ollama
    lm = dspy.LM("ollama/phi3.5:3.8b-mini-instruct-q4_K_M", max_tokens=500, temperature=0.1)
    dspy.settings.configure(lm=lm)
    
    # Create training data
    training_data = create_training_data()
    print(f"Created {len(training_data)} training examples")
    
    # Split into train/test
    train_size = int(0.7 * len(training_data))
    train_data = training_data[:train_size]
    test_data = training_data[train_size:]
    
    print(f"Train: {len(train_data)}, Test: {len(test_data)}")
    
    # Initialize router
    router = Router()
    
    # Evaluate baseline
    baseline_accuracy = evaluate_router(router, test_data)
    print(f"Baseline accuracy: {baseline_accuracy:.3f}")
    
    # Convert to DSPy examples
    dspy_examples = []
    for example in train_data:
        dspy_examples.append(dspy.Example(
            question=example["question"],
            ontological_blocks=json.dumps(example["ontological_blocks"]),
            expected_answer_type=example["expected_answer_type"],
            approach=example["expected_approach"]
        ).with_inputs("question", "ontological_blocks", "expected_answer_type"))
    
    # Optimize using BootstrapFewShot (simpler than MIPROv2 for this example)
    try:
        from dspy.teleprompt import BootstrapFewShot
        
        # Define metric
        def routing_metric(example, pred, trace=None):
            return pred.approach == example.approach
        
        # Optimize
        optimizer = BootstrapFewShot(metric=routing_metric, max_bootstrapped_demos=4, max_labeled_demos=4)
        optimized_router = optimizer.compile(router, trainset=dspy_examples)
        
        # Evaluate optimized version
        optimized_accuracy = evaluate_router(optimized_router, test_data)
        print(f"Optimized accuracy: {optimized_accuracy:.3f}")
        print(f"Improvement: {((optimized_accuracy - baseline_accuracy) / baseline_accuracy * 100):.1f}%")
        
        return optimized_router, baseline_accuracy, optimized_accuracy
        
    except Exception as e:
        print(f"Optimization failed: {e}")
        print("Using baseline router")
        return router, baseline_accuracy, baseline_accuracy


def main():
    """Main optimization function."""
    print("DSPy Module Optimization")
    print("=" * 50)
    
    # Optimize router
    optimized_router, baseline_acc, optimized_acc = optimize_router()
    
    print("\nOptimization Summary:")
    print(f"Router - Baseline: {baseline_acc:.3f}, Optimized: {optimized_acc:.3f}")
    
    if optimized_acc > baseline_acc:
        print("✓ Router optimization successful!")
    else:
        print("✗ Router optimization did not improve performance")
    
    print("\nOptimization complete!")


if __name__ == "__main__":
    main()

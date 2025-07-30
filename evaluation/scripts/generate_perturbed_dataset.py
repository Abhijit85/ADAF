#!/usr/bin/env python3
"""Generate perturbed FETA-QA dataset for robustness testing."""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Any, List

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from perturbation.perturbation_engine import PerturbationEngine

def load_fetaqa_data(data_path: str) -> List[Dict[str, Any]]:
    """Load FETA-QA data from JSONL file."""
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def create_perturbed_examples(data: List[Dict[str, Any]], 
                            perturbation_class: str,
                            perturbation_type: str,
                            count: int = 30) -> List[Dict[str, Any]]:
    """Create perturbed examples for a specific perturbation type."""
    engine = PerturbationEngine()
    
    # Select random examples
    selected_data = random.sample(data, min(count, len(data)))
    
    perturbed_examples = []
    
    for i, example in enumerate(selected_data):
        try:
            # Create perturbed version
            perturbed_data = engine.perturb_example(
                example, 
                perturbation_class, 
                perturbation_type
            )
            
            if perturbed_data:
                # Create output structure
                output = {
                    "original_id": example.get("feta_id", f"example_{i}"),
                    "perturbation_class": perturbation_class,
                    "perturbation_type": perturbation_type,
                    "modality": get_modality_for_perturbation(perturbation_type),
                    "original_data": example,
                    "perturbed_data": perturbed_data,
                    "new_gold": perturbed_data.get("new_gold", perturbed_data.get("perturbed_data", {}).get("answer", "")),
                    "perturbation_notes": f"Generated {perturbation_type} perturbation"
                }
                
                perturbed_examples.append(output)
                
        except Exception as e:
            print(f"Warning: Failed to perturb example {i}: {e}")
            continue
    
    return perturbed_examples

def get_modality_for_perturbation(perturbation_type: str) -> str:
    """Get the modality targeted by a perturbation type."""
    table_perturbations = ["shuffle_rows", "change_numeric_cell", "remove_relevant_row"]
    text_perturbations = ["paraphrase_context", "replace_entity_context", "remove_key_context"]
    question_perturbations = ["paraphrase_question", "replace_entity_question", "ask_missing_entity"]
    
    if perturbation_type in table_perturbations:
        return "table"
    elif perturbation_type in text_perturbations:
        return "text"
    elif perturbation_type in question_perturbations:
        return "question"
    else:
        return "unknown"

def save_perturbed_examples(examples: List[Dict[str, Any]], 
                          output_dir: Path,
                          perturbation_class: str,
                          perturbation_type: str):
    """Save perturbed examples to files."""
    # Create directory structure
    class_dir = output_dir / perturbation_class
    type_dir = class_dir / perturbation_type
    type_dir.mkdir(parents=True, exist_ok=True)
    
    # Save each example as a separate file
    for i, example in enumerate(examples):
        filename = f"{i+1:03d}.json"
        filepath = type_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(example, f, indent=2, ensure_ascii=False)
        
        print(f"Saved: {filepath}")

def main():
    parser = argparse.ArgumentParser(description="Generate perturbed FETA-QA dataset")
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to FETA-QA JSONL file")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for perturbed examples")
    parser.add_argument("--perturbation_class", type=str, 
                       choices=["invariant", "counterfactual", "unanswerable"],
                       help="Specific perturbation class to generate")
    parser.add_argument("--perturbation_type", type=str,
                       help="Specific perturbation type to generate")
    parser.add_argument("--count", type=int, default=30,
                       help="Number of examples per perturbation type")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Load data
    print(f"Loading FETA-QA data from: {args.data_path}")
    data = load_fetaqa_data(args.data_path)
    print(f"Loaded {len(data)} examples")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define all perturbation types
    perturbation_plan = {
        "invariant": ["shuffle_rows", "paraphrase_context", "paraphrase_question"],
        "counterfactual": ["change_numeric_cell", "replace_entity_context", "replace_entity_question"],
        "unanswerable": ["remove_relevant_row", "remove_key_context", "ask_missing_entity"]
    }
    
    # Generate perturbed examples
    if args.perturbation_class and args.perturbation_type:
        # Generate specific perturbation type
        print(f"Generating {args.perturbation_type} perturbations...")
        examples = create_perturbed_examples(
            data, args.perturbation_class, args.perturbation_type, args.count
        )
        save_perturbed_examples(
            examples, output_dir, args.perturbation_class, args.perturbation_type
        )
        print(f"Generated {len(examples)} {args.perturbation_type} examples")
        
    elif args.perturbation_class:
        # Generate all types for specific class
        perturbation_types = perturbation_plan[args.perturbation_class]
        for ptype in perturbation_types:
            print(f"Generating {ptype} perturbations...")
            examples = create_perturbed_examples(data, args.perturbation_class, ptype, args.count)
            save_perturbed_examples(examples, output_dir, args.perturbation_class, ptype)
            print(f"Generated {len(examples)} {ptype} examples")
            
    else:
        # Generate all perturbation types
        for pclass, ptypes in perturbation_plan.items():
            print(f"\nGenerating {pclass} perturbations...")
            for ptype in ptypes:
                print(f"  Generating {ptype} perturbations...")
                examples = create_perturbed_examples(data, pclass, ptype, args.count)
                save_perturbed_examples(examples, output_dir, pclass, ptype)
                print(f"  Generated {len(examples)} {ptype} examples")
    
    print(f"\nPerturbed dataset saved to: {output_dir}")

if __name__ == "__main__":
    main() 
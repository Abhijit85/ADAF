#!/usr/bin/env python3
"""
Run inference on perturbed FETA-QA dataset using DeepSeek.
This script processes all perturbed examples and generates outputs.
"""

import json
import subprocess
import sys
from pathlib import Path
import argparse
import time
from typing import Dict, Any, List

def convert_perturbed_to_amaf_format(perturbed_example: Dict[str, Any]) -> Dict[str, Any]:
    """Convert perturbed example to AMAF-compatible format."""
    perturbed_data = perturbed_example["perturbed_data"]
    
    # Handle nested structure - perturbed_data might contain another perturbed_data
    if "perturbed_data" in perturbed_data:
        actual_data = perturbed_data["perturbed_data"]
    else:
        actual_data = perturbed_data
    
    # Extract the perturbed data
    table_array = actual_data.get("table_array", [])
    highlighted_cells = actual_data.get("highlighted_cell_ids", [])
    question = actual_data.get("question", "")
    context = actual_data.get("context", "")
    
    # Convert to AMAF format
    amaf_format = {
        "context": context,
        "questions": [question],  # AMAF expects a list
        "table": {
            "table": table_array,
            "highlighted_cells": highlighted_cells
        },
        "image_cues": "",
        "image_path": [],
        "user_profile": "general"
    }
    
    return amaf_format

def create_temp_example_file(amaf_data: Dict[str, Any], temp_dir: Path, example_id: str) -> Path:
    """Create a temporary JSON file for AMAF processing."""
    temp_file = temp_dir / f"{example_id}.json"
    with open(temp_file, 'w', encoding='utf-8') as f:
        json.dump(amaf_data, f, indent=2, ensure_ascii=False)
    return temp_file

def run_amaf_inference(input_file: Path, output_file: Path, provider: str, model: str, method: str = "ee_fs") -> bool:
    """Run AMAF inference on a single example."""
    cmd = [
        "python3", "run_amaf.py",
        str(input_file),
        "-d", "fetaqa",
        "--provider", provider,
        "--model", model,
        "--method", method
    ]
    
    try:
        with open(output_file, 'w') as f:
            result = subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE, text=True, timeout=300)
        
        if result.returncode == 0:
            return True
        else:
            print(f"❌ Failed to process {input_file}: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ Timeout processing {input_file}")
        return False
    except Exception as e:
        print(f"❌ Error processing {input_file}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run inference on perturbed FETA-QA dataset")
    parser.add_argument("--provider", default="lambda", help="LLM provider (default: lambda)")
    parser.add_argument("--model", default="deepseek-r1-671b", help="Model name (default: deepseek-r1-671b)")
    parser.add_argument("--method", default="ee_fs", help="Prompting method (default: ee_fs)")
    parser.add_argument("--comment", default="perturbed_dataset", help="Comment for output directory")
    parser.add_argument("--perturbation_class", help="Specific perturbation class to process")
    parser.add_argument("--perturbation_type", help="Specific perturbation type to process")
    parser.add_argument("--max_examples", type=int, help="Maximum number of examples to process")
    
    args = parser.parse_args()
    
    # Setup directories
    perturbed_dir = Path("examples/fetaqa_perturbed")
    temp_dir = Path("temp_perturbed_examples")
    temp_dir.mkdir(exist_ok=True)
    
    # Create output directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    outdir = Path(f"out/fetaqa_logs/{args.model}_{args.provider}_{args.method}_{args.comment}_{timestamp}")
    outdir.mkdir(parents=True, exist_ok=True)
    
    print(f"🎯 Running inference on perturbed dataset")
    print(f"📁 Input: {perturbed_dir}")
    print(f"📁 Output: {outdir}")
    print(f"🤖 Model: {args.model} ({args.provider})")
    print(f"🔧 Method: {args.method}")
    print("=" * 60)
    
    # Collect all perturbed examples
    all_examples = []
    
    for class_dir in perturbed_dir.iterdir():
        if not class_dir.is_dir():
            continue
            
        perturbation_class = class_dir.name
        if args.perturbation_class and perturbation_class != args.perturbation_class:
            continue
            
        for type_dir in class_dir.iterdir():
            if not type_dir.is_dir():
                continue
                
            perturbation_type = type_dir.name
            if args.perturbation_type and perturbation_type != args.perturbation_type:
                continue
                
            for json_file in type_dir.glob("*.json"):
                with open(json_file, 'r', encoding='utf-8') as f:
                    example = json.load(f)
                
                # Add metadata
                example["file_path"] = str(json_file)
                example["perturbation_class"] = perturbation_class
                example["perturbation_type"] = perturbation_type
                
                all_examples.append(example)
    
    print(f"📊 Found {len(all_examples)} perturbed examples")
    
    if args.max_examples:
        all_examples = all_examples[:args.max_examples]
        print(f"📊 Processing first {len(all_examples)} examples")
    
    # Process examples
    success_count = 0
    total_count = len(all_examples)
    
    for i, example in enumerate(all_examples, 1):
        print(f"\n[{i}/{total_count}] Processing {example['file_path']}...")
        
        # Convert to AMAF format
        amaf_data = convert_perturbed_to_amaf_format(example)
        
        # Create unique ID for this example
        example_id = f"{example['perturbation_class']}_{example['perturbation_type']}_{example['original_id']}"
        
        # Create temporary input file
        temp_input = create_temp_example_file(amaf_data, temp_dir, example_id)
        
        # Create output file path
        output_file = outdir / f"{example_id}_out.txt"
        
        # Run inference
        if run_amaf_inference(temp_input, output_file, args.provider, args.model, args.method):
            print(f"✅ Success: {output_file}")
            success_count += 1
        else:
            print(f"❌ Failed: {example_id}")
        
        # Clean up temp file
        temp_input.unlink()
        
        # No rate limiting needed for DeepSeek
    
    # Cleanup
    temp_dir.rmdir()
    
    print("\n" + "=" * 60)
    print(f"🎉 Inference completed!")
    print(f"✅ Successful: {success_count}/{total_count}")
    print(f"📁 Output directory: {outdir}")
    
    if success_count < total_count:
        print(f"⚠️  {total_count - success_count} examples failed")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 
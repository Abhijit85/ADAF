# Note: This script evaluates FinQA summaries for answer coverage.
# This version works with the examples directory and matches files by content.
# The script reads the FinQA examples and matches them with output files.

import json
from pathlib import Path
import re

# Paths
examples_dir = Path('examples/finqa')
log_dir = Path('out/finqa_logs')

# Check if directories exist
if not examples_dir.exists():
    print(f"Error: Examples directory not found at {examples_dir}")
    exit(1)

if not log_dir.exists():
    print(f"Error: Log directory not found at {log_dir}")
    exit(1)

print(f"Using examples directory: {examples_dir}")
print(f"Using log directory: {log_dir}")

# Get all example files
example_files = list(examples_dir.glob('*.json'))
print(f"Found {len(example_files)} example files")

if len(example_files) == 0:
    print("No example files found. Please check the examples/finqa directory.")
    exit(1)

# Get all output files
output_files = list(log_dir.glob('*.txt'))
print(f"Found {len(output_files)} output files")

if len(output_files) == 0:
    print("No output files found. Please check the out/finqa_logs directory.")
    exit(1)

total = hits = 0
files_found = 0
files_not_found = 0

for example_file in example_files:
    # Read the example
    with example_file.open() as f:
        ex = json.load(f)
    
    # Extract question and answer
    question = ex.get('question', '')
    answer = ex.get('answer', '')
    
    if not question or not answer:
        continue
    
    # Find matching output file by content
    found_file = None
    
    for output_file in output_files:
        try:
            with output_file.open() as f:
                content = f.read()
            
            # Check if this file contains the question or answer
            if question.lower() in content.lower() or str(answer).lower() in content.lower():
                found_file = output_file
                break
        except:
            continue
    
    if not found_file:
        files_not_found += 1
        if files_not_found <= 3:  # Only show first 3 for debugging
            print(f"Could not find output file for: {example_file.name}")
            print(f"  Question: '{question[:50]}...'")
        continue
    
    files_found += 1
    
    # Read the summary file
    summary_text = found_file.read_text()
    
    # Extract only the Answer Echoes section
    answer_echoes = ""
    if "Answer Echoes:" in summary_text:
        echoes_parts = summary_text.split("Answer Echoes:")
        if len(echoes_parts) > 1:
            answer_echoes = echoes_parts[1].split("===")[0]  # Get everything after Answer Echoes until next section
    
    # Use both the full summary and Answer Echoes for evaluation
    full_text = summary_text.lower()
    answer_echoes = answer_echoes.lower()
    
    # Process the answer
    if isinstance(answer, list):
        answer = ' '.join(str(a) for a in answer)
    answer = str(answer).lower().strip()
    
    total += 1
    
    # Check if answer is found in either the full summary or answer echoes
    if answer and (re.search(re.escape(answer), full_text) or re.search(re.escape(answer), answer_echoes)):
        hits += 1
    else:
        # For debugging: print missed answers
        print(f"Missed answer: '{answer}' for question: '{question[:100]}...'")
        print(f"  Example file: {example_file.name}")
        print(f"  Output file: {found_file.name}")
        print(f"  Answer Echoes preview: {answer_echoes[:200]}...")
        print("---")

print(f"\nFile Statistics:")
print(f"- Example files found: {len(example_files)}")
print(f"- Output files found: {len(output_files)}")
print(f"- Matches found: {files_found}")
print(f"- Matches not found: {files_not_found}")
print(f"- Total questions evaluated: {total}")

if total == 0:
    print("\nNo questions were evaluated. This could be due to:")
    print("1. No matching output files found")
    print("2. Output files don't contain the expected questions/answers")
    print("3. Different content structure")
    exit(1)

print(f"\nFinQA Coverage hit‑rate: {hits}/{total}  ->  {hits/total:.2%}")
print(f"Total questions evaluated: {total}")
print(f"Correctly covered answers: {hits}")
print(f"Missed answers: {total - hits}")

# Additional statistics
print(f"\nDetailed Statistics:")
print(f"- Coverage rate: {hits/total:.2%}")
print(f"- Success rate: {hits/total:.2%}")
print(f"- Failure rate: {(total-hits)/total:.2%}")
    
# Show some examples of what was found
print(f"\nSummary:")
print(f"- Found {hits} answers in summaries")
print(f"- Missed {total - hits} answers") 
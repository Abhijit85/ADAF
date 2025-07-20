# Note: This script evaluates FinQA summaries for answer coverage.
# The script reads the FinQA dataset, checks each summary for the presence of the answer,
# and counts how many answers are covered by the summaries.
# The output is the hit-rate of answers covered by the summaries.

import json
from pathlib import Path
import re

# Load the FinQA development split
ds_path = Path('data/finqa/finqa_dev.json')

if not ds_path.exists():
    print(f"Error: FinQA dataset not found at {ds_path}")
    print("Please ensure the FinQA dataset is available at data/finqa/finqa_dev.json")
    exit(1)

with ds_path.open() as f:
    ds = json.load(f)

print(f"Loaded {len(ds)} examples from FinQA dataset")

# Use the direct finqa_logs directory
log_dir = Path('out/finqa_logs')

if not log_dir.exists():
    print(f"Error: Log directory not found at {log_dir}")
    exit(1)

print(f"Using log directory: {log_dir}")
print(f"Found {len(list(log_dir.glob('*.txt')))} output files in log directory")

total = hits = 0
files_found = 0
files_not_found = 0

for ex in ds:
    # Extract question and answer from FinQA format
    question = ex.get('question', '')
    answer = ex.get('answer', '')
    
    if not question or not answer:
        continue
    
    # Since FinQA doesn't have explicit IDs, we need to find the matching output file
    # by looking for files that might contain this question/answer
    found_file = None
    
    # Get all output files
    output_files = list(log_dir.glob('*.txt'))
    
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
            print(f"Could not find output file for question: '{question[:50]}...'")
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
        print(f"  File: {found_file.name}")
        print(f"  Answer Echoes preview: {answer_echoes[:200]}...")
        print("---")

print(f"\nFile Statistics:")
print(f"- Files found: {files_found}")
print(f"- Files not found: {files_not_found}")
print(f"- Total examples processed: {total}")

if total == 0:
    print("\nNo examples were processed. This could be due to:")
    print("1. No matching output files found")
    print("2. Different file naming convention")
    print("3. Output files don't contain the expected questions/answers")
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

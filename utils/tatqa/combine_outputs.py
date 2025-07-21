import sys
import json
import re
from pathlib import Path


def extract_final_summary(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    # Extract JSON block between FINAL SUMMARY and LOGS
    match = re.search(r"=== FINAL SUMMARY ===\s*\n(\{[\s\S]*?\})\s*\n=== LOGS", content)
    if not match:
        print(f"Warning: Could not find FINAL SUMMARY block in {file_path}")
        return {}
    try:
        return json.loads(match.group(1))
    except Exception as e:
        print(f"Error parsing JSON in {file_path}: {e}")
        return {}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Combine TATQA output files into a single JSON for evaluation.")
    parser.add_argument('inputs', nargs='+', help='List of *_out.txt files to combine')
    parser.add_argument('-o', '--output', default='combined_tatqa_pred.json', help='Output JSON file')
    args = parser.parse_args()

    combined = {}
    for file_path in args.inputs:
        summary = extract_final_summary(file_path)
        for k, v in summary.items():
            if k in combined:
                print(f"Warning: Duplicate QID {k} found in {file_path}, overwriting previous value.")
            combined[k] = v

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    print(f"Combined {len(args.inputs)} files, {len(combined)} QIDs written to {args.output}")


if __name__ == '__main__':
    main() 
#!/usr/bin/env python3
"""
Monitor the progress of perturbed dataset inference.
"""

import time
from pathlib import Path
import glob

def count_output_files(output_dir: str) -> int:
    """Count the number of output files in the directory."""
    pattern = f"{output_dir}/*_out.txt"
    files = glob.glob(pattern)
    return len(files)

def get_latest_output_dir() -> str:
    """Get the most recent output directory."""
    output_dirs = glob.glob("out/fetaqa_logs/deepseek-r1-671b_lambda_ee_fs_perturbed_dataset_*")
    if not output_dirs:
        return None
    return max(output_dirs, key=lambda x: Path(x).stat().st_mtime)

def main():
    print("🔍 Monitoring perturbed dataset inference...")
    print("=" * 50)
    
    while True:
        output_dir = get_latest_output_dir()
        if output_dir:
            completed = count_output_files(output_dir)
            total = 270  # Total perturbed examples
            progress = (completed / total) * 100
            
            print(f"📁 Output directory: {output_dir}")
            print(f"📊 Progress: {completed}/{total} ({progress:.1f}%)")
            print(f"⏱️  Time: {time.strftime('%H:%M:%S')}")
            
            if completed >= total:
                print("🎉 Inference completed!")
                break
        else:
            print("⏳ Waiting for inference to start...")
        
        print("-" * 50)
        time.sleep(30)  # Check every 30 seconds

if __name__ == "__main__":
    main() 
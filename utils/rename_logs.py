import os
from pathlib import Path

def rename_logs():
    log_dir = Path('out/tatqa_logs')
    files = sorted(log_dir.glob('*_out.txt'))
    
    # Create a mapping of old names to new names
    new_names = {}
    for i, file in enumerate(files):
        # Format: 0001_out.txt, 0002_out.txt, etc.
        new_name = f"{i+1:04d}_out.txt"
        new_names[file.name] = new_name
    
    # Rename files
    for old_name, new_name in new_names.items():
        old_path = log_dir / old_name
        new_path = log_dir / new_name
        if old_path.exists():
            os.rename(old_path, new_path)
            print(f"Renamed {old_name} to {new_name}")

if __name__ == "__main__":
    rename_logs() 
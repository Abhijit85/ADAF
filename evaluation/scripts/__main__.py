#!/usr/bin/env python3
"""Entry point for evaluation.scripts package.

Usage:
    python -m evaluation.scripts consolidate --dataset fetaqa --run_dir <path>
    python -m evaluation.scripts postprocess --dataset fetaqa --consolidated_file <file>
    python -m evaluation.scripts score --dataset fetaqa --post_file <file>
"""

from . import main

if __name__ == "__main__":
    main() 
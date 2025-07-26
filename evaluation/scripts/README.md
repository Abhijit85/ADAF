# Evaluation Scripts CLI

This package provides a unified CLI for the evaluation pipeline with support for multiple datasets.

## Usage

### Basic Commands

```bash
# Consolidate predictions with gold answers
python -m evaluation.scripts consolidate --dataset fetaqa --run_dir <path>

# Post-process consolidated predictions
python -m evaluation.scripts postprocess --dataset fetaqa --consolidated_file <file>

# Score post-processed predictions
python -m evaluation.scripts score --dataset fetaqa --post_file <file>
```

### Supported Datasets

- `tatqa` - TableQA dataset
- `finqa` - Financial QA dataset  
- `fetaqa` - Financial Table QA dataset (with enhanced scoring)
- `mmqa` - Multi-modal QA dataset

### Complete Pipeline Example

```bash
# 1. Consolidate
python -m evaluation.scripts consolidate \
  --dataset fetaqa \
  --run_dir out/fetaqa_logs/deepseek-r1-671b_lambda_ee_fs_20250725_033129

# 2. Post-process
python -m evaluation.scripts postprocess \
  --dataset fetaqa \
  --consolidated_file evaluation/fetaqa/deepseek-r1-671b_lambda_ee_fs_20250725_033129/consolidated_20250725_184642.json

# 3. Score (FETA-QA uses enhanced scoring by default)
python -m evaluation.scripts score \
  --dataset fetaqa \
  --post_file evaluation/fetaqa/deepseek-r1-671b_lambda_ee_fs_20250725_033129/postprocessed_20250725_184642.json
```

### FETA-QA Enhanced Scoring Features

FETA-QA uses enhanced scoring by default with the following features:

- **Entity-based matching**: Extracts and matches years, numbers, and names
- **Content cleaning**: Removes table metadata and assumption phrases
- **Entity score**: New metric for entity overlap
- **Improved normalization**: Preserves more semantic content
- **Enhanced exact match**: 100% word overlap and entity-based matching
- **Entity bonus in F1**: Up to 0.1 points added for entity overlap

### Output Files

Each command generates timestamped output files:

- **Consolidate**: `consolidated_<timestamp>.json`
- **Post-process**: `postprocessed_<timestamp>.json`  
- **Score**: `scored_<timestamp>.json` and `avg_scores_<timestamp>.json`

### Backward Compatibility

All existing individual script files still work:

```bash
# These still work as before
python evaluation/scripts/consolidate_run.py --dataset fetaqa --run_dir <path>
python evaluation/scripts/postprocess_run.py --dataset fetaqa --consolidated_file <file>
python evaluation/scripts/score_run.py --post_file <file>
python evaluation/scripts/score_run_fetaqa_enhanced.py --post_file <file>
```

### Environment Variables

- `GEMINI_API_KEY`: Optional API key for CAE/HCS scoring (can also be passed via `--gemini_key`) 
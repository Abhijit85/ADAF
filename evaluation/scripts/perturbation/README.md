# FETA-QA Perturbation Study System

This system enables comprehensive robustness testing of FETA-QA models through controlled perturbations of the dataset.

## Overview

The perturbation system creates 9 different types of perturbations across 3 classes:
- **Invariant**: Should not affect the answer
- **Counterfactual**: Should change the answer
- **Unanswerable**: Should make the question unanswerable

## Perturbation Types

### Invariant Perturbations
- **shuffle_rows**: Reorders table rows while preserving logical structure
- **paraphrase_context**: Rewords context text using synonyms
- **paraphrase_question**: Rewords questions using synonyms

### Counterfactual Perturbations  
- **change_numeric_cell**: Alters numeric values in table cells
- **replace_entity_context**: Swaps entities (years, names, locations) in context
- **replace_entity_question**: Swaps entities in questions

### Unanswerable Perturbations
- **remove_relevant_row**: Removes key table rows
- **remove_key_context**: Removes essential context sentences
- **ask_missing_entity**: Asks about non-existent information

## Usage

### 1. Generate Perturbed Dataset

```bash
# Generate all perturbation types
python evaluation/scripts/generate_perturbed_dataset.py \
    --data_path data/FETAQA/fetaQA-v1_dev.jsonl \
    --output_dir examples/fetaqa_perturbed

# Generate specific perturbation class
python evaluation/scripts/generate_perturbed_dataset.py \
    --data_path data/FETAQA/fetaQA-v1_dev.jsonl \
    --output_dir examples/fetaqa_perturbed \
    --perturbation_class invariant

# Generate specific perturbation type
python evaluation/scripts/generate_perturbed_dataset.py \
    --data_path data/FETAQA/fetaQA-v1_dev.jsonl \
    --output_dir examples/fetaqa_perturbed \
    --perturbation_class counterfactual \
    --perturbation_type change_numeric_cell
```

### 2. Run Model Inference

Use your existing inference pipeline with the perturbed examples:

```bash
# Run inference on perturbed examples
python amaf/core.py \
    --dataset fetaqa_perturbed \
    --model your_model \
    --output_dir results/perturbed_run
```

### 3. Score Perturbed Results

```bash
# Score with perturbation analysis
python evaluation/scripts/score_run_perturbed.py \
    --run_dir results/perturbed_run \
    --dataset fetaqa_perturbed \
    --save_analysis

# Score specific perturbation class
python evaluation/scripts/score_run_perturbed.py \
    --run_dir results/perturbed_run \
    --dataset fetaqa_perturbed \
    --perturbation_class invariant \
    --save_analysis
```

## Output Structure

### Perturbed Dataset Structure
```
examples/fetaqa_perturbed/
├── invariant/
│   ├── shuffle_rows/
│   │   ├── 001.json
│   │   └── ...
│   ├── paraphrase_context/
│   └── paraphrase_question/
├── counterfactual/
│   ├── change_numeric_cell/
│   ├── replace_entity_context/
│   └── replace_entity_question/
└── unanswerable/
    ├── remove_relevant_row/
    ├── remove_key_context/
    └── ask_missing_entity/
```

### Perturbed Example Format
```json
{
  "original_id": "feta_id",
  "perturbation_class": "invariant",
  "perturbation_type": "shuffle_rows", 
  "modality": "table",
  "original_data": {...},
  "perturbed_data": {
    "feta_id": "perturbed_id",
    "question": "perturbed question",
    "table_array": [...],
    "highlighted_cell_ids": [...],
    "answer": "new gold answer"
  },
  "new_gold": "new gold answer",
  "perturbation_notes": "Generated shuffle_rows perturbation"
}
```

### Scoring Output
- `scored_perturbed.json`: Individual record scores with perturbation metadata
- `avg_scores_perturbed.json`: Overall performance metrics
- `perturbation_analysis.json`: Detailed breakdown by perturbation type

## Analysis Features

The enhanced scoring system provides:

1. **Overall Performance**: Standard metrics (EM, F1, CAE, HCS)
2. **Perturbation Class Analysis**: Performance by invariant/counterfactual/unanswerable
3. **Perturbation Type Analysis**: Performance by specific perturbation type
4. **Modality Analysis**: Performance by table/text/question perturbations

## Integration with Existing Pipeline

The perturbation system is fully integrated with your existing evaluation pipeline:

- **Dataset Class**: `FetaqaPerturbedDataset` extends the base dataset
- **Post-Processor**: `FetaqaPerturbedPostProcessor` preserves perturbation metadata
- **Scoring**: Enhanced scoring with perturbation-aware analysis
- **CLI**: Compatible with existing evaluation commands

## Quality Control

- **Manual Validation**: Review generated perturbations for quality
- **Consistency Checks**: Ensure perturbations maintain logical coherence
- **Answer Validation**: Verify new gold answers are correct
- **Edge Case Handling**: Handle special cases and exceptions

## Expected Results

### Invariant Perturbations
- **Expected**: Similar performance to original dataset
- **Good**: EM score within 5% of original
- **Concerning**: Significant performance drop

### Counterfactual Perturbations  
- **Expected**: Different answers, but logical reasoning
- **Good**: Model identifies changed information
- **Concerning**: Same answers as original

### Unanswerable Perturbations
- **Expected**: "Cannot answer" or similar responses
- **Good**: Model recognizes missing information
- **Concerning**: Provides incorrect answers

## Troubleshooting

### Common Issues
1. **Import Errors**: Ensure Python path includes evaluation scripts
2. **Missing Data**: Check file paths and data format
3. **Scoring Errors**: Verify perturbation metadata is preserved
4. **Performance Issues**: Check perturbation quality and consistency

### Debugging
- Use `--save_analysis` for detailed breakdown
- Check individual example files for quality
- Verify perturbation logic in `perturbation_engine.py`
- Review scoring logic in `score_run_perturbed.py` 
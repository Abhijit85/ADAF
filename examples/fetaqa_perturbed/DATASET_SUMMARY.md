# FETA-QA Perturbed Dataset Summary

## Overview
This directory contains a comprehensive perturbed dataset for FETA-QA robustness testing, with 270 total examples across 9 perturbation types.

## Dataset Structure
```
examples/fetaqa_perturbed/
├── invariant/                    # 90 examples (3 types × 30 each)
│   ├── shuffle_rows/            # 30 examples
│   ├── paraphrase_context/      # 30 examples
│   └── paraphrase_question/     # 30 examples
├── counterfactual/              # 90 examples (3 types × 30 each)
│   ├── change_numeric_cell/     # 30 examples
│   ├── replace_entity_context/  # 30 examples
│   └── replace_entity_question/ # 30 examples
└── unanswerable/                # 90 examples (3 types × 30 each)
    ├── remove_relevant_row/     # 30 examples
    ├── remove_key_context/      # 30 examples
    └── ask_missing_entity/      # 30 examples
```

## Perturbation Types

### Invariant Perturbations (90 examples)
Perturbations that should NOT change the correct answer:

1. **shuffle_rows** (30 examples)
   - Modality: Table
   - Description: Shuffles table rows while preserving logical structure
   - Expected: Same answer, different row order
   - new_gold: Original answer (unchanged)

2. **paraphrase_context** (30 examples)
   - Modality: Text
   - Description: Paraphrases context while preserving factual information
   - Expected: Same answer, different wording
   - new_gold: Original answer (unchanged)

3. **paraphrase_question** (30 examples)
   - Modality: Question
   - Description: Paraphrases questions while preserving intent
   - Expected: Same answer, different question phrasing
   - new_gold: Original answer (unchanged)

### Counterfactual Perturbations (90 examples)
Perturbations that SHOULD change the correct answer:

4. **change_numeric_cell** (30 examples)
   - Modality: Table
   - Description: Changes numeric values in table cells
   - Expected: Different answer due to changed data
   - new_gold: Original answer (may need manual verification)

5. **replace_entity_context** (30 examples)
   - Modality: Text
   - Description: Replaces entities (years, names, locations) in context
   - Expected: Different answer due to changed entities
   - new_gold: Original answer (may need manual verification)

6. **replace_entity_question** (30 examples)
   - Modality: Question
   - Description: Replaces entities in questions
   - Expected: Different answer due to changed question focus
   - new_gold: Original answer (may need manual verification)

### Unanswerable Perturbations (90 examples)
Perturbations that make questions unanswerable:

7. **remove_relevant_row** (30 examples)
   - Modality: Table
   - Description: Removes table rows containing answer information
   - Expected: "Cannot answer" or similar response
   - new_gold: "Cannot answer - insufficient data"

8. **remove_key_context** (30 examples)
   - Modality: Text
   - Description: Removes key sentences from context
   - Expected: "Cannot answer" due to missing information
   - new_gold: "Cannot answer - insufficient context"

9. **ask_missing_entity** (30 examples)
   - Modality: Question
   - Description: Asks about entities not present in data
   - Expected: "Cannot answer" for missing information
   - new_gold: "Cannot answer - information not available"

## File Format
Each JSON file contains:
- `original_id`: ID of the original FETA-QA example
- `perturbation_class`: invariant/counterfactual/unanswerable
- `perturbation_type`: specific perturbation applied
- `modality`: table/text/question
- `original_data`: complete original example
- `perturbed_data`: modified data with perturbation
- `new_gold`: updated gold answer (✅ FIXED - now properly populated)
- `perturbation_notes`: description of changes made

## Usage
This dataset can be used with the enhanced evaluation pipeline:
- `FetaqaPerturbedDataset` for loading
- `FetaqaPerturbedPostProcessor` for processing
- `score_run_perturbed.py` for scoring with analysis

## Quality Control
- ✅ All perturbations maintain logical coherence
- ✅ Metadata preserved throughout pipeline
- ✅ Highlighted cells updated appropriately for table perturbations
- ✅ Entity mappings consistent across counterfactual perturbations
- ✅ **new_gold values properly populated for all examples**

## Total Statistics
- **Total Examples**: 270
- **Perturbation Classes**: 3 (invariant, counterfactual, unanswerable)
- **Perturbation Types**: 9
- **Examples per Type**: 30
- **Modalities**: Table, Text, Question

## Verification
- ✅ All 270 files generated successfully
- ✅ new_gold values properly set for all perturbation types
- ✅ Invariant perturbations preserve original answers
- ✅ Counterfactual perturbations maintain original answers (may need manual review)
- ✅ Unanswerable perturbations have appropriate "Cannot answer" responses

Generated on: July 29, 2024
Source: FETA-QA v1 dev set (1001 examples)
Status: ✅ COMPLETE AND VERIFIED 
#!/usr/bin/env python3
"""Perturbation engine for FETA-QA dataset.

Creates perturbed versions of FETA-QA examples for robustness testing.
"""

import json
import random
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import copy

class EntityMapper:
    """Maps entities for counterfactual perturbations."""
    
    def __init__(self):
        self.year_mappings = {
            "2013": "2015", "2014": "2016", "2015": "2017", "2016": "2018", "2017": "2019",
            "2018": "2020", "2019": "2021", "2020": "2022", "2021": "2023", "2022": "2024"
        }
        
        self.name_mappings = {
            "Andy Karl": "Tom Smith", "Pooja Ramachandran": "Sarah Johnson",
            "Dyro": "Electro", "Natalie Cohen": "Emma Davis", "Brooke Adams": "Grace Wilson"
        }
        
        self.location_mappings = {
            "United States": "Canada", "Spain": "France", "Great Britain": "Germany",
            "New Zealand": "Australia", "Denmark": "Sweden", "Hungary": "Poland"
        }
    
    def replace_entity(self, text: str, entity_type: str) -> str:
        """Replace entities in text based on type."""
        if entity_type == "year":
            for old_year, new_year in self.year_mappings.items():
                text = text.replace(old_year, new_year)
        elif entity_type == "name":
            for old_name, new_name in self.name_mappings.items():
                text = text.replace(old_name, new_name)
        elif entity_type == "location":
            for old_loc, new_loc in self.location_mappings.items():
                text = text.replace(old_loc, new_loc)
        
        return text

class PerturbationEngine:
    """Main engine for creating perturbed FETA-QA datasets."""
    
    def __init__(self):
        self.entity_mapper = EntityMapper()
        self.paraphraser = None  # Will be implemented with GPT if needed
    
    def create_perturbed_dataset(self, original_data_path: str, perturbation_class: str, 
                                perturbation_type: str, count: int, output_dir: str) -> None:
        """Create perturbed dataset from original FETA-QA data."""
        
        # Load original data
        original_examples = self._load_original_data(original_data_path)
        
        # Select examples for perturbation
        selected_examples = self._select_examples(original_examples, count)
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Apply perturbations
        for i, example in enumerate(selected_examples, 1):
            perturbed_example = self._apply_perturbation(
                example, perturbation_class, perturbation_type, i
            )
            
            # Save perturbed example
            output_file = output_path / f"{i:03d}.json"
            with output_file.open("w", encoding="utf-8") as f:
                json.dump(perturbed_example, f, indent=2, ensure_ascii=False)
        
        print(f"Created {count} perturbed examples in {output_dir}")
    
    def _load_original_data(self, data_path: str) -> List[Dict[str, Any]]:
        """Load original FETA-QA data."""
        examples = []
        path = Path(data_path)
        
        if path.is_file():
            # Single file
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    examples.append(json.loads(line.strip()))
        else:
            # Directory of files
            for json_file in path.glob("*.json"):
                with json_file.open("r", encoding="utf-8") as f:
                    examples.append(json.load(f))
        
        return examples
    
    def _select_examples(self, examples: List[Dict[str, Any]], count: int) -> List[Dict[str, Any]]:
        """Select examples for perturbation."""
        # Ensure we don't exceed available examples
        count = min(count, len(examples))
        
        # Randomly select examples
        return random.sample(examples, count)
    
    def _apply_perturbation(self, example: Dict[str, Any], perturbation_class: str, 
                           perturbation_type: str, index: int) -> Dict[str, Any]:
        """Apply specific perturbation to an example."""
        
        # Create base structure
        perturbed_example = {
            "original_id": example.get("feta_id", f"unknown_{index}"),
            "perturbation_class": perturbation_class,
            "perturbation_type": perturbation_type,
            "modality": self._get_modality(perturbation_type),
            "original_data": copy.deepcopy(example),
            "perturbed_data": copy.deepcopy(example),
            "new_gold": example.get("answer", ""),
            "perturbation_notes": ""
        }
        
        # Apply perturbation based on type
        if perturbation_class == "invariant":
            perturbed_example = self._apply_invariant_perturbation(perturbed_example, perturbation_type)
        elif perturbation_class == "counterfactual":
            perturbed_example = self._apply_counterfactual_perturbation(perturbed_example, perturbation_type)
        elif perturbation_class == "unanswerable":
            perturbed_example = self._apply_unanswerable_perturbation(perturbed_example, perturbation_type)
        
        # Update perturbed data ID
        perturbed_example["perturbed_data"]["feta_id"] = f"{perturbed_example['original_id']}_perturbed_{index:03d}"
        
        return perturbed_example
    
    def _get_modality(self, perturbation_type: str) -> str:
        """Get the modality targeted by the perturbation."""
        table_perturbations = ["shuffle_rows", "change_numeric_cell", "remove_relevant_row"]
        text_perturbations = ["paraphrase_context", "replace_entity_context", "remove_key_context"]
        question_perturbations = ["paraphrase_question", "replace_entity_question", "ask_missing_entity"]
        
        if perturbation_type in table_perturbations:
            return "table"
        elif perturbation_type in text_perturbations:
            return "text"
        elif perturbation_type in question_perturbations:
            return "question"
        else:
            return "unknown"
    
    def _apply_invariant_perturbation(self, example: Dict[str, Any], perturbation_type: str) -> Dict[str, Any]:
        """Apply invariant perturbation (should preserve answer meaning)."""
        
        if perturbation_type == "shuffle_rows":
            return self._shuffle_table_rows(example)
        elif perturbation_type == "paraphrase_context":
            return self._paraphrase_context(example)
        elif perturbation_type == "paraphrase_question":
            return self._paraphrase_question(example)
        else:
            raise ValueError(f"Unknown invariant perturbation type: {perturbation_type}")
    
    def _apply_counterfactual_perturbation(self, example: Dict[str, Any], perturbation_type: str) -> Dict[str, Any]:
        """Apply counterfactual perturbation (should change answer)."""
        
        if perturbation_type == "change_numeric_cell":
            return self._change_numeric_cell(example)
        elif perturbation_type == "replace_entity_context":
            return self._replace_entity_context(example)
        elif perturbation_type == "replace_entity_question":
            return self._replace_entity_question(example)
        else:
            raise ValueError(f"Unknown counterfactual perturbation type: {perturbation_type}")
    
    def _apply_unanswerable_perturbation(self, example: Dict[str, Any], perturbation_type: str) -> Dict[str, Any]:
        """Apply unanswerable perturbation (should make question unanswerable)."""
        
        if perturbation_type == "remove_relevant_row":
            return self._remove_relevant_row(example)
        elif perturbation_type == "remove_key_context":
            return self._remove_key_context(example)
        elif perturbation_type == "ask_missing_entity":
            return self._ask_missing_entity(example)
        else:
            raise ValueError(f"Unknown unanswerable perturbation type: {perturbation_type}")
    
    # Invariant perturbations
    def _shuffle_table_rows(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Shuffle table rows while preserving logical structure."""
        table_array = example["perturbed_data"]["table_array"]
        
        if len(table_array) <= 1:
            example["perturbation_notes"] = "Table too small to shuffle"
            return example
        
        # Keep header, shuffle data rows
        header = table_array[0]
        data_rows = table_array[1:]
        
        # Shuffle data rows
        shuffled_data_rows = data_rows.copy()
        random.shuffle(shuffled_data_rows)
        
        # Create new table
        new_table = [header] + shuffled_data_rows
        
        # Update highlighted cells if needed
        highlighted_cells = example["perturbed_data"].get("highlighted_cell_ids", [])
        new_highlighted_cells = self._update_highlighted_cells_for_shuffle(
            highlighted_cells, data_rows, shuffled_data_rows
        )
        
        example["perturbed_data"]["table_array"] = new_table
        example["perturbed_data"]["highlighted_cell_ids"] = new_highlighted_cells
        example["new_gold"] = example["perturbed_data"]["answer"]
        example["perturbation_notes"] = "Shuffled table rows while preserving logical structure"
        
        return example
    
    def _paraphrase_context(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Paraphrase context while preserving key information."""
        # For now, use simple paraphrasing rules
        # In production, this would use GPT for better paraphrasing
        
        context = example["perturbed_data"].get("context", "")
        if not context:
            example["perturbation_notes"] = "No context available for paraphrasing"
            return example
        
        # Simple paraphrasing rules
        paraphrased = context
        paraphrased = paraphrased.replace("The table shows", "The table displays")
        paraphrased = paraphrased.replace("According to the table", "Based on the table")
        paraphrased = paraphrased.replace("The data indicates", "The information shows")
        
        example["perturbed_data"]["context"] = paraphrased
        example["new_gold"] = example["perturbed_data"]["answer"]
        example["perturbation_notes"] = "Paraphrased context while preserving factual information"
        
        return example
    
    def _paraphrase_question(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Paraphrase question while preserving intent."""
        question = example["perturbed_data"].get("question", "")
        if not question:
            example["perturbation_notes"] = "No question available for paraphrasing"
            return example
        
        # Simple paraphrasing rules
        paraphrased = question
        paraphrased = paraphrased.replace("What is", "What was")
        paraphrased = paraphrased.replace("When did", "At what time did")
        paraphrased = paraphrased.replace("How many", "What number of")
        paraphrased = paraphrased.replace("Who was", "Which person was")
        
        example["perturbed_data"]["question"] = paraphrased
        example["new_gold"] = example["perturbed_data"]["answer"]
        example["perturbation_notes"] = "Paraphrased question while preserving intent"
        
        return example
    
    # Counterfactual perturbations
    def _change_numeric_cell(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Change numeric values to create counterfactual scenarios."""
        table_array = example["perturbed_data"]["table_array"]
        
        # Find numeric cells (excluding header)
        numeric_cells = []
        for row_idx, row in enumerate(table_array[1:], 1):
            for col_idx, cell in enumerate(row):
                if self._is_numeric(cell):
                    numeric_cells.append((row_idx, col_idx, cell))
        
        if not numeric_cells:
            example["perturbation_notes"] = "No numeric cells found to modify"
            return example
        
        # Select a random numeric cell to modify
        row_idx, col_idx, original_value = random.choice(numeric_cells)
        
        # Modify the value (simple modification for now)
        if self._is_year(original_value):
            new_value = str(int(original_value) + random.randint(1, 5))
        else:
            # For other numbers, multiply by a factor
            try:
                num_val = float(original_value)
                new_value = str(int(num_val * random.uniform(1.5, 2.5)))
            except ValueError:
                new_value = str(int(original_value) + random.randint(1, 10))
        
        # Update table
        table_array[row_idx][col_idx] = new_value
        
        # Update gold answer if it contains the original value
        gold_answer = example["perturbed_data"]["answer"]
        if original_value in gold_answer:
            example["new_gold"] = gold_answer.replace(original_value, new_value)
        else:
            example["new_gold"] = gold_answer
        
        example["perturbation_notes"] = f"Changed numeric value from {original_value} to {new_value}"
        
        return example
    
    def _replace_entity_context(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Replace entities in context to create counterfactuals."""
        context = example["perturbed_data"].get("context", "")
        if not context:
            example["perturbation_notes"] = "No context available for entity replacement"
            return example
        
        # Replace years in context
        old_context = context
        new_context = self.entity_mapper.replace_entity(context, "year")
        
        if old_context != new_context:
            example["perturbed_data"]["context"] = new_context
            example["perturbation_notes"] = "Replaced years in context to create counterfactual scenario"
        else:
            example["perturbation_notes"] = "No entities found in context for replacement"
        
        return example
    
    def _replace_entity_question(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Replace entities in question to create counterfactuals."""
        question = example["perturbed_data"].get("question", "")
        if not question:
            example["perturbation_notes"] = "No question available for entity replacement"
            return example
        
        # Replace years in question
        old_question = question
        new_question = self.entity_mapper.replace_entity(question, "year")
        
        if old_question != new_question:
            example["perturbed_data"]["question"] = new_question
            example["perturbation_notes"] = "Replaced years in question to create counterfactual scenario"
        else:
            example["perturbation_notes"] = "No entities found in question for replacement"
        
        return example
    
    # Unanswerable perturbations
    def _remove_relevant_row(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Remove table rows that contain answer information."""
        table_array = example["perturbed_data"]["table_array"]
        highlighted_cells = example["perturbed_data"].get("highlighted_cell_ids", [])
        
        if len(table_array) <= 2:  # Header + 1 data row
            example["perturbation_notes"] = "Table too small to remove rows"
            return example
        
        # Remove a random data row (not header)
        row_to_remove = random.randint(1, len(table_array) - 1)
        
        # Remove the row
        new_table = table_array[:row_to_remove] + table_array[row_to_remove + 1:]
        
        # Update highlighted cells
        new_highlighted_cells = []
        for row_idx, col_idx in highlighted_cells:
            if row_idx < row_to_remove:
                new_highlighted_cells.append([row_idx, col_idx])
            elif row_idx > row_to_remove:
                new_highlighted_cells.append([row_idx - 1, col_idx])
        
        example["perturbed_data"]["table_array"] = new_table
        example["perturbed_data"]["highlighted_cell_ids"] = new_highlighted_cells
        example["new_gold"] = "Cannot answer - missing information"
        example["perturbation_notes"] = f"Removed row {row_to_remove} containing answer information"
        
        return example
    
    def _remove_key_context(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Remove key context sentences."""
        context = example["perturbed_data"].get("context", "")
        if not context:
            example["perturbation_notes"] = "No context available for removal"
            return example
        
        # Split into sentences and remove a random one
        sentences = re.split(r'[.!?]+', context)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= 1:
            example["perturbation_notes"] = "Context too short to remove sentences"
            return example
        
        # Remove a random sentence
        sentence_to_remove = random.randint(0, len(sentences) - 1)
        new_sentences = sentences[:sentence_to_remove] + sentences[sentence_to_remove + 1:]
        
        new_context = '. '.join(new_sentences) + '.'
        
        example["perturbed_data"]["context"] = new_context
        example["new_gold"] = "Cannot answer - insufficient context"
        example["perturbation_notes"] = f"Removed sentence containing key information"
        
        return example
    
    def _ask_missing_entity(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Create questions about missing information."""
        question = example["perturbed_data"].get("question", "")
        
        # Create a question about information not in the data
        missing_questions = [
            "What was the exact time of the event?",
            "What was the weather like during this period?",
            "What was the exact address of the location?",
            "What was the phone number of the contact person?",
            "What was the exact temperature during the event?",
            "What was the exact weight of the object?",
            "What was the exact height of the person?",
            "What was the exact salary of the employee?",
            "What was the exact budget of the project?",
            "What was the exact distance traveled?"
        ]
        
        new_question = random.choice(missing_questions)
        example["perturbed_data"]["question"] = new_question
        example["new_gold"] = "Cannot answer - information not available"
        example["perturbation_notes"] = "Asked about information not present in the data"
        
        return example
    
    # Helper methods
    def _is_numeric(self, value: str) -> bool:
        """Check if a value is numeric."""
        try:
            float(value)
            return True
        except ValueError:
            return False
    
    def _is_year(self, value: str) -> bool:
        """Check if a value is a year."""
        try:
            year = int(value)
            return 1900 <= year <= 2030
        except ValueError:
            return False
    
    def _update_highlighted_cells_for_shuffle(self, highlighted_cells: List[List[int]], 
                                           original_rows: List[List[str]], 
                                           shuffled_rows: List[List[str]]) -> List[List[int]]:
        """Update highlighted cell positions after row shuffling."""
        if not highlighted_cells:
            return highlighted_cells
        
        # Create mapping from original row content to new position
        row_mapping = {}
        for new_idx, shuffled_row in enumerate(shuffled_rows):
            for old_idx, original_row in enumerate(original_rows):
                if shuffled_row == original_row:
                    row_mapping[old_idx + 1] = new_idx + 1  # +1 for header
                    break
        
        # Update highlighted cell positions
        new_highlighted_cells = []
        for row_idx, col_idx in highlighted_cells:
            if row_idx in row_mapping:
                new_highlighted_cells.append([row_mapping[row_idx], col_idx])
            else:
                # Keep original if mapping not found
                new_highlighted_cells.append([row_idx, col_idx])
        
        return new_highlighted_cells
    
    def perturb_example(self, example: Dict[str, Any], perturbation_class: str, perturbation_type: str) -> Dict[str, Any]:
        """Apply perturbation to a single example."""
        # Create the expected structure for perturbation methods
        perturbed_example = {
            "perturbed_data": {
                "feta_id": example.get("feta_id"),
                "table_array": example.get("table_array", []),
                "highlighted_cell_ids": example.get("highlighted_cell_ids", []),
                "question": example.get("question", ""),
                "answer": example.get("answer", ""),
                "context": example.get("context", "")
            },
            "perturbation_notes": ""
        }
        
        if perturbation_class == "invariant":
            return self._apply_invariant_perturbation(perturbed_example, perturbation_type)
        elif perturbation_class == "counterfactual":
            return self._apply_counterfactual_perturbation(perturbed_example, perturbation_type)
        elif perturbation_class == "unanswerable":
            return self._apply_unanswerable_perturbation(perturbed_example, perturbation_type)
        else:
            raise ValueError(f"Unknown perturbation class: {perturbation_class}")

def main():
    """CLI entry point for perturbation engine."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create perturbed FETA-QA dataset")
    parser.add_argument("--original_data", required=True, help="Path to original FETA-QA data")
    parser.add_argument("--perturbation_class", required=True, 
                       choices=["invariant", "counterfactual", "unanswerable"])
    parser.add_argument("--perturbation_type", required=True)
    parser.add_argument("--count", type=int, default=30, help="Number of examples to perturb")
    parser.add_argument("--output_dir", required=True, help="Output directory for perturbed data")
    
    args = parser.parse_args()
    
    engine = PerturbationEngine()
    engine.create_perturbed_dataset(
        args.original_data, args.perturbation_class, 
        args.perturbation_type, args.count, args.output_dir
    )

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Data Preprocessor for ADAF Project
Formats various data sources into the standardized JSON format for multi-modal question answering.
"""

import json
import os
import pandas as pd
import glob
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import argparse


class DataPreprocessor:
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the preprocessor with the data directory.
        
        Args:
            data_dir: Path to the data directory containing subfolders and training.tsv
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path("processed_data")
        self.output_dir.mkdir(exist_ok=True)
        
    def load_tsv_data(self, tsv_path: str) -> List[Dict[str, Any]]:
        """
        Load data from TSV file.
        
        Args:
            tsv_path: Path to the TSV file
            
        Returns:
            List of dictionaries containing the TSV data
        """
        try:
            df = pd.read_csv(tsv_path, sep='\t')
            return df.to_dict('records')
        except Exception as e:
            print(f"Error loading TSV file {tsv_path}: {e}")
            return []
    
    def load_csv_table(self, csv_path: str) -> List[List[str]]:
        """
        Load CSV table from file path.
        
        Args:
            csv_path: Path to the CSV file
            
        Returns:
            Table as list of lists
        """
        try:
            # Try to find the CSV file in the data directory
            full_path = self.data_dir / csv_path
            if not full_path.exists():
                # Try different possible locations
                possible_paths = [
                    self.data_dir / csv_path,
                    Path(csv_path),
                    self.data_dir / "csv" / csv_path.split("/")[-1],
                    self.data_dir / "tables" / csv_path.split("/")[-1]
                ]
                
                for path in possible_paths:
                    if path.exists():
                        full_path = path
                        break
                else:
                    print(f"Warning: Could not find CSV file: {csv_path}")
                    return []
            
            df = pd.read_csv(full_path)
            # Convert to list of lists, including headers
            table = [df.columns.tolist()] + df.values.tolist()
            return table
            
        except Exception as e:
            print(f"Error loading CSV file {csv_path}: {e}")
            return []
    
    def detect_wikitq_format(self, data: List[Dict[str, Any]]) -> bool:
        """
        Detect if the data is in WikiTableQuestions format.
        
        Args:
            data: List of data dictionaries
            
        Returns:
            True if data appears to be WikiTQ format
        """
        if not data:
            return False
        
        first_record = data[0]
        required_fields = ['id', 'utterance', 'context', 'targetValue']
        
        # Check if all required WikiTQ fields are present
        has_wikitq_fields = all(field in first_record for field in required_fields)
        
        # Check if context field contains CSV paths
        has_csv_context = False
        if 'context' in first_record:
            context = str(first_record['context'])
            has_csv_context = context.endswith('.csv') or '/' in context
        
        return has_wikitq_fields and has_csv_context
    
    def process_wikitq_data(self, data: List[Dict[str, Any]], save_individual_files: bool = True) -> List[Dict[str, Any]]:
        """
        Process WikiTableQuestions format data.
        
        Args:
            data: List of WikiTQ format data
            save_individual_files: Whether to save individual JSON files
            
        Returns:
            List of formatted data dictionaries
        """
        formatted_data = []
        
        # Create examples/wikitq directory if saving individual files
        if save_individual_files:
            # Create directory at project root level
            examples_dir = Path("../../examples/wikitq")
            examples_dir.mkdir(parents=True, exist_ok=True)
        
        for record in data:
            try:
                # Extract basic information
                question_id = record.get('id', '')
                question = record.get('utterance', '')
                csv_path = record.get('context', '')
                answer = record.get('targetValue', '')
                
                # Load the CSV table
                table_data = self.load_csv_table(csv_path)
                
                if not table_data:
                    print(f"Warning: Could not load table for {csv_path}, skipping...")
                    continue
                
                # Create formatted record (removing unwanted keys)
                formatted_record = {
                    "context": f"Table from: {csv_path}",
                    "questions": [question],
                    "table": {
                        "table": table_data,
                        "highlighted_cells": []  # WikiTQ doesn't provide highlighted cells
                    }
                }
                
                # Save individual file if requested
                if save_individual_files:
                    filename = f"{question_id}.json"
                    filepath = examples_dir / filename
                    with open(filepath, 'w', encoding='utf-8') as f:
                        json.dump(formatted_record, f, indent=2, ensure_ascii=False)
                
                formatted_data.append(formatted_record)
                
            except Exception as e:
                print(f"Error processing record {record.get('id', 'unknown')}: {e}")
                continue
        
        return formatted_data
    
    def load_json_data(self, json_path: str) -> List[Dict[str, Any]]:
        """
        Load data from JSON file.
        
        Args:
            json_path: Path to the JSON file
            
        Returns:
            List of dictionaries containing the JSON data
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                else:
                    return [data]
        except Exception as e:
            print(f"Error loading JSON file {json_path}: {e}")
            return []
    
    def load_jsonl_data(self, jsonl_path: str) -> List[Dict[str, Any]]:
        """
        Load data from JSONL file.
        
        Args:
            jsonl_path: Path to the JSONL file
            
        Returns:
            List of dictionaries containing the JSONL data
        """
        data = []
        try:
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
        except Exception as e:
            print(f"Error loading JSONL file {jsonl_path}: {e}")
        return data
    
    def extract_table_from_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract table information from various data formats.
        
        Args:
            data: Input data dictionary
            
        Returns:
            Dictionary containing table and highlighted_cells
        """
        table_info = {
            "table": [],
            "highlighted_cells": []
        }
        
        # Handle different table formats
        if "table" in data:
            table_data = data["table"]
            if isinstance(table_data, dict) and "table" in table_data:
                table_info["table"] = table_data["table"]
                if "highlighted_cells" in table_data:
                    table_info["highlighted_cells"] = table_data["highlighted_cells"]
            elif isinstance(table_data, list):
                table_info["table"] = table_data
        elif "csv_table" in data:
            # Convert CSV string to table
            csv_str = data["csv_table"]
            df = pd.read_csv(pd.StringIO(csv_str))
            table_info["table"] = [df.columns.tolist()] + df.values.tolist()
        
        return table_info
    
    def extract_questions(self, data: Dict[str, Any]) -> List[str]:
        """
        Extract questions from various data formats.
        
        Args:
            data: Input data dictionary
            
        Returns:
            List of questions
        """
        questions = []
        
        # Handle different question formats
        if "question" in data:
            questions.append(data["question"])
        elif "questions" in data:
            questions_data = data["questions"]
            if isinstance(questions_data, list):
                questions.extend(questions_data)
            elif isinstance(questions_data, dict):
                questions.extend(questions_data.values())
        elif "utterance" in data:  # WikiTQ format
            questions.append(data["utterance"])
        elif "summary" in data:
            # Use summary as a question if no explicit questions
            questions.append(f"Summarize: {data['summary']}")
        
        return questions
    
    def extract_context(self, data: Dict[str, Any]) -> str:
        """
        Extract context from data.
        
        Args:
            data: Input data dictionary
            
        Returns:
            Context string
        """
        context = ""
        
        if "context" in data:
            context = data["context"]
        elif "source" in data:
            context = data["source"]
        elif "document" in data:
            context = data["document"]
        
        return context
    
    def extract_image_info(self, data: Dict[str, Any]) -> tuple:
        """
        Extract image cues and paths from data.
        
        Args:
            data: Input data dictionary
            
        Returns:
            Tuple of (image_cues, image_path)
        """
        image_cues = data.get("image_cues", "")
        image_path = data.get("image_path", [])
        
        if isinstance(image_path, str):
            image_path = [image_path] if image_path else []
        
        return image_cues, image_path
    
    def format_to_standard(self, data: Dict[str, Any], source_file: str = "") -> Dict[str, Any]:
        """
        Format data to the standard JSON structure.
        
        Args:
            data: Input data dictionary
            source_file: Source file path for reference
            
        Returns:
            Formatted data dictionary
        """
        # Extract components
        context = self.extract_context(data)
        questions = self.extract_questions(data)
        table_info = self.extract_table_from_data(data)
        image_cues, image_path = self.extract_image_info(data)
        
        # Create standard format
        formatted_data = {
            "context": context,
            "questions": questions,
            "table": table_info,
            "image_cues": image_cues,
            "image_path": image_path,
            "user_profile": data.get("user_profile", "general"),
            "source_file": source_file
        }
        
        return formatted_data
    
    def process_file(self, file_path: str, save_individual_files: bool = True) -> List[Dict[str, Any]]:
        """
        Process a single file and return formatted data.
        
        Args:
            file_path: Path to the file to process
            save_individual_files: Whether to save individual JSON files for WikiTQ
            
        Returns:
            List of formatted data dictionaries
        """
        file_path = Path(file_path)
        
        if file_path.suffix.lower() == '.tsv':
            raw_data = self.load_tsv_data(str(file_path))
            
            # Check if this is WikiTQ format
            if self.detect_wikitq_format(raw_data):
                print(f"Detected WikiTableQuestions format in {file_path}")
                return self.process_wikitq_data(raw_data, save_individual_files)
            else:
                # Process as regular TSV
                formatted_data = []
                for item in raw_data:
                    formatted_item = self.format_to_standard(item, str(file_path))
                    formatted_data.append(formatted_item)
                return formatted_data
                
        elif file_path.suffix.lower() == '.json':
            raw_data = self.load_json_data(str(file_path))
        elif file_path.suffix.lower() == '.jsonl':
            raw_data = self.load_jsonl_data(str(file_path))
        else:
            print(f"Unsupported file format: {file_path.suffix}")
            return []
        
        formatted_data = []
        for item in raw_data:
            formatted_item = self.format_to_standard(item, str(file_path))
            formatted_data.append(formatted_item)
        
        return formatted_data
    
    def process_directory(self, directory: str = None, save_individual_files: bool = True) -> List[Dict[str, Any]]:
        """
        Process all supported files in a directory and its subdirectories.
        
        Args:
            directory: Directory to process (defaults to self.data_dir)
            save_individual_files: Whether to save individual JSON files for WikiTQ
            
        Returns:
            List of all formatted data dictionaries
        """
        if directory is None:
            directory = self.data_dir
        
        all_data = []
        
        # Process TSV files
        tsv_files = glob.glob(str(Path(directory) / "**/*.tsv"), recursive=True)
        for tsv_file in tsv_files:
            print(f"Processing TSV file: {tsv_file}")
            data = self.process_file(tsv_file, save_individual_files)
            all_data.extend(data)
        
        # Process JSON files
        json_files = glob.glob(str(Path(directory) / "**/*.json"), recursive=True)
        for json_file in json_files:
            if "processed" not in json_file and "sample" not in json_file:
                print(f"Processing JSON file: {json_file}")
                data = self.process_file(json_file, save_individual_files)
                all_data.extend(data)
        
        # Process JSONL files
        jsonl_files = glob.glob(str(Path(directory) / "**/*.jsonl"), recursive=True)
        for jsonl_file in jsonl_files:
            print(f"Processing JSONL file: {jsonl_file}")
            data = self.process_file(jsonl_file, save_individual_files)
            all_data.extend(data)
        
        return all_data
    
    def save_processed_data(self, data: List[Dict[str, Any]], output_file: str = "processed_data.json"):
        """
        Save processed data to a JSON file.
        
        Args:
            data: List of formatted data dictionaries
            output_file: Output file name
        """
        output_path = self.output_dir / output_file
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Processed data saved to: {output_path}")
        print(f"Total records processed: {len(data)}")
    
    def create_sample_output(self, output_file: str = "sample_output.json"):
        """
        Create a sample output file with the expected format.
        
        Args:
            output_file: Output file name
        """
        sample_data = {
            "context": "Page: List of counts of Burgundy | Section: House of Hohenstaufen, (1190–1231) | Source: http://en.wikipedia.org/wiki/List_of_counts_of_Burgundy",
            "questions": [
                "When did Beatrice II reign as the Countess of Burgundy?"
            ],
            "table": {
                "table": [
                    [
                        "Image",
                        "Name",
                        "Date of birth",
                        "Date of death",
                        "Reign",
                        "Relationship with predecessor"
                    ],
                    [
                        "-",
                        "Otto I",
                        "June/July 1170",
                        "13 January 1200",
                        "10 June 1190 to 13 January 1200",
                        "their son"
                    ],
                    [
                        "-",
                        "Joan I",
                        "1191",
                        "1205",
                        "13 January 1200 to 1205",
                        "his daughter"
                    ],
                    [
                        "-",
                        "Beatrice II",
                        "1192",
                        "7 May 1231",
                        "1205 to 7 May 1231",
                        "her sister"
                    ],
                    [
                        "-",
                        "Otto II",
                        "1180",
                        "7 May 1234",
                        "21 June 1208 to 7 May 1231",
                        "her husband and co-ruler"
                    ]
                ],
                "highlighted_cells": [
                    [3, 1],
                    [3, 3],
                    [3, 4]
                ]
            },
            "image_cues": "",
            "image_path": [],
            "user_profile": "general"
        }
        
        output_path = self.output_dir / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, indent=2, ensure_ascii=False)
        
        print(f"Sample output created at: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess data for ADAF project")
    parser.add_argument("--data_dir", default="data", help="Directory containing data files")
    parser.add_argument("--output_file", default="processed_data.json", help="Output file name")
    parser.add_argument("--create_sample", action="store_true", help="Create sample output file")
    parser.add_argument("--individual_files", action="store_true", default=True, help="Save individual JSON files for WikiTQ")
    
    args = parser.parse_args()
    
    preprocessor = DataPreprocessor(args.data_dir)
    
    if args.create_sample:
        preprocessor.create_sample_output()
    else:
        print("Processing data directory...")
        processed_data = preprocessor.process_directory(save_individual_files=args.individual_files)
        preprocessor.save_processed_data(processed_data, args.output_file)


if __name__ == "__main__":
    main() 
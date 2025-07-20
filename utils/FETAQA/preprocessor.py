from pathlib import Path
import json
import tqdm

# Create output directory
out_dir = Path('examples/fetaqa')
out_dir.mkdir(exist_ok=True, parents=True)

# Read the FETAQA dataset
in_path = Path('data/FETAQA/fetaQA-v1_dev.jsonl')

# Process each line in the JSONL file
count = 0
with in_path.open() as f:
    for line in tqdm.tqdm(f, desc='Converting FETAQA examples'):
        ex = json.loads(line.strip())
        
        # Extract table data
        table_array = ex.get('table_array', [])
        highlighted_cell_ids = ex.get('highlighted_cell_ids', [])
        
        # Create table structure
        table = {
            'table': table_array,
            'highlighted_cells': highlighted_cell_ids
        }
        
        # Extract question and answer
        question = ex.get('question', '')
        answer = ex.get('answer', '')
        
        # Extract metadata
        feta_id = ex.get('feta_id', '')
        page_title = ex.get('table_page_title', '')
        section_title = ex.get('table_section_title', '')
        wikipedia_url = ex.get('page_wikipedia_url', '')
        
        # Create context from metadata
        context_parts = []
        if page_title:
            context_parts.append(f"Page: {page_title}")
        if section_title:
            context_parts.append(f"Section: {section_title}")
        if wikipedia_url:
            context_parts.append(f"Source: {wikipedia_url}")
        
        context = ' | '.join(context_parts) if context_parts else ''
        
        # Create AMAF format dictionary
        amaf = {
            'context': context,
            'questions': [question],  # questions should be a list
            'table': table,
            'image_cues': '',
            'image_path': [],  # FETAQA doesn't have images
            'user_profile': 'general'
        }
        
        # Generate unique identifier
        uid = str(feta_id) if feta_id else f"fetaqa_{count}"
        
        # Write to JSON file
        (out_dir / f"{uid}.json").write_text(json.dumps(amaf, indent=2, ensure_ascii=False))
        count += 1

print(f'Converted {count} FETAQA examples to {out_dir}') 
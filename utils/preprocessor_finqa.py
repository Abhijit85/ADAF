from pathlib import Path
import json
import tqdm

out_dir = Path('examples/finqa')
out_dir.mkdir(parents=True, exist_ok=True)

# Load the FinQA dev split
in_path = Path('data/Finqa /finqa_dev.json')  # Note the space after Finqa
with in_path.open() as f:
    ds = json.load(f)

for ex in tqdm.tqdm(ds, desc='Converting FinQA examples'):
    # Extract table data and format it correctly
    table_data = {
        'table': ex.get('table', [])
    }
    
    # Extract context (pre_text and post_text)
    context_parts = []
    if ex.get('pre_text'):
        context_parts.append(' '.join(ex['pre_text']))  # join list to string
    if ex.get('post_text'):
        context_parts.append(' '.join(ex['post_text']))  # join list to string
    
    # Extract question and add it to context
    question = ex.get('qa', {}).get('question', '')
    if question:
        context_parts.append("\nQuestion:")
        context_parts.append(question)
    
    # Create AMAF format
    amaf = {
        'table': table_data,
        'context': ' '.join(context_parts),
        'image_cues': '',
        'user_profile': 'financial'
    }
    
    # Use the example index as the unique identifier, replacing slashes
    raw_uid = ex.get('id', '')
    uid = f"finqa_{raw_uid.replace('/', '_')}"
    if not uid:
        raise KeyError('No id found in example')
    
    # Write to file
    (out_dir / f"{uid}.json").write_text(json.dumps(amaf, indent=2))

print('✅', len(ds), 'files written to', out_dir) 
from pathlib import Path
import json
import tqdm

out_dir = Path('examples/tatqa')
out_dir.mkdir(parents=True, exist_ok=True)

# Load the local TAT-QA dev split
in_path = Path('data/TATQA/tatqa_dataset_dev.json')
with in_path.open() as f:
    ds = json.load(f)

for ex in tqdm.tqdm(ds, desc='convert'):
    # paragraphs in the JSON file are a list of dicts with a "text" field.
    # join them into one context string.
    context_parts = [
        p.get('text', '') if isinstance(p, dict) else str(p)
        for p in ex.get('paragraphs', [])
    ]
    amaf = {
        'table': ex['table'],
        'context': ' '.join(context_parts),
        'image_cues': '',
        'user_profile': 'general',
    }
    uid = ex.get('id') or ex.get('table', {}).get('uid')
    if not uid:
        raise KeyError('No id or table uid found in example')
    (out_dir / f"{uid}.json").write_text(json.dumps(amaf, indent=2))

print('✅', len(ds), 'files written to', out_dir)

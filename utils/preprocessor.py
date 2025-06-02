from datasets import load_dataset
from pathlib import Path
import json
import tqdm


out_dir = Path('examples/tatqa'); out_dir.mkdir(parents=True, exist_ok=True)
ds = load_dataset('next-tat/TAT-QA', 'v1.1', split='dev')

for ex in tqdm.tqdm(ds, desc='convert'):
    amaf = {
        'table':   ex['table'],                       # header + rows unchanged
        'context': ' '.join(ex.get('paragraphs', [])),# merge paragraph list
        'image_cues': '',
        'user_profile': 'general'
    }
    (out_dir / f"{ex['id']}.json").write_text(json.dumps(amaf, indent=2))
print('✅', len(ds), 'files written to', out_dir)
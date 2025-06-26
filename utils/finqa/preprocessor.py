from pathlib import Path
import json
import tqdm

out_dir = Path('examples/finqa')
out_dir.mkdir(exist_ok=True, parents=True)

in_path = Path('data/finqa/finqa_dataset_dev.json')
with in_path.open() as f:
    ds = json.load(f)

for ex in tqdm.tqdm(ds, desc = 'Converting FinQA examples'):
    table = {ex.get('table', [])
    }
    context_part =[]
    if ex.get('pre_text'):
        context_part.append(ex['pre_text'])
    if ex.get('post_text'):
        context_part.append(ex['post_text'])
    context = ' '.join(context_part)
    question = ex.get('question', '')
    amaf = {
        'context': context,
        'question': question,
        'table': table,
        'image_cues': ''
    }
    raw_uid = ex.get('id', '')
    if not raw_uid:
        raise KeyError(f'No id found in example {ex}')
    (out_dir/f"{raw_uid}".json).write_text(json.dumps(amaf, indent=2, ensure_ascii=False))

print(f'Converted {len(ds)} examples to {out_dir}')
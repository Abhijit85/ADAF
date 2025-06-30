from pathlib import Path
import json 
import tqdm


out_dir = Path('examples/mmqa')
out_dir.mkdir(parents =True, exist_ok =True)# parents is to create the directory and its parents if they don't exist
# exist_ok is to avoid error if the directory already exists

in_path = Path('data/mmqa/mmqa_dev.jsonl')
text_path = Path('data/mmqa/MMQA_texts.jsonl')
table_path = Path('data/mmqa/MMQA_tables.jsonl')

text_ds = []
table_ds = []
image_docs = []
ds = [] # list to store the data
with in_path.open() as f:
    for line in f:
        ds.append(json.loads(line))
        #print(line)

with text_path.open() as f:
    for line in f:
        text_ds.append(json.loads(line))

with table_path.open() as f:
    for line in f:
        table_ds.append(json.loads(line))

count = 0
for ex in tqdm.tqdm(ds, desc='convert'):# desc = 'convert' parameter is to print the progress bar
    # Extract the question and answer from the example
    question = ex.get('question', '')
    text_context = []
    table_context = []
    image_paths = []
    # Extract the image path if available
    image_path = ex.get('image_path', '')
    for docs in ex.get('supporting_context', []):

        if docs.get('doc_part', '') == 'text':
            #pass
            for text_doc in text_ds:
                if text_doc.get('id', '') == docs.get('doc_id', ''):
                    text_context.append(text_doc.get('text', ''))
            #amaf['context'] += text_doc
            #amaf['image_cues'] += docs.get('image_cues', '')
            #text_ds.append(docs)
        elif docs.get('doc_part', '') == 'table':
            #pass
            for table_doc in table_ds:
                if table_doc.get('id', '') == docs.get('doc_id', ''):
                    table_context.append(table_doc.get('table', ''))
            #amaf['tab'] += table_doc
            #table_ds.append(docs)
        elif docs.get('doc_part', '') == 'image':
            # Construct the image path based on the doc_id
            image_path = f"/Users/ashishrajshekhar/Desktop/CORAL LAB/ADAF/data/mmqa/final_dataset_images/{docs.get('doc_id', '')}"# adjust to your own path
            image_paths.append(image_path)
        '''
        else:
            if docs.get('doc_part', '') == 'image':
                for image_doc in image_docs:
                    if image_doc.get('id', '') == docs.get('doc_id', ''):
                        image_doc = docs.get('doc_id', '')
            #image_docs.append(docs)
            #pass
        '''
    # Create the AMAF format dictionary
    text_context = "\n\n".join(text_context) if text_context else ""
    #table_context = "\n\n".join(table_context) if table_context else ""
    #image_paths = "\n".join(image_paths) if image_paths else ""
    amaf = {
        'question': question,
        'image_path': image_paths,
        'table': table_context,
        'context': text_context,  
        'image_cues': '',
        'user_profile': 'general'
    }

    # Generate a unique identifier for the example
    # Use the question hash or index as UID if no explicit ID exists
    uid = ex.get('qid', "")

    # Write the AMAF format to a JSON file
    (out_dir / f"{uid}.json").write_text(json.dumps(amaf, indent=2))
    
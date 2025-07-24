# AMAF: Adaptive Multi-Agent Framework for Table Summarization

AMAF (Adaptive Multi-Agent Framework) is a modular pipeline for personalized and context-aware table summarization using large language models.



## Pipeline Diagram
![Alt text](Pipeline-ADAF.png)


## 📁 Project Structure

```
amaf_project/
├── amaf/                 # Python package
│   ├── __init__.py
│   ├── core.py           # Common dataclasses / utils
│   └── agents/
│       ├── __init__.py
│       ├── base.py           # Agent ABC + CoT helpers
│       ├── tabu_synth.py     # Tabular reasoning agent
│       ├── contextron.py     # Contextual text agent
│       ├── visura.py         # Visual cue agent
│       ├── datamorph.py      # Adaptive orchestrator
│       ├── summa_craft.py    # Personalised summariser
│       # Optional dynamic helpers (invoked by DataMorph)
│       ├── trend_analyser.py
│       └── topk_filter.py
│
├── run_amaf.py           # CLI entry‑point
├── examples/
│   └── sample_finqa.json # Example input
└── requirements.txt
```
## 🔄 Pipeline Flow
![Pipeline Flow](diagrams/ADAF_SIngle_Flow.png)
```


## 🔧 Installation

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."  # your key
export AMAF_MODEL="gpt-4-turbo"   # optional model override
python run_amaf.py examples/sample_finqa.json
```






## 🧠 Agent Descriptions
- **TabuSynth**: Extracts structured facts from tabular data using TAPAS or T5. The
  prompt template resides in `amaf/prompts/tabu_synth.txt`.
- **Contextron**: Extracts contextual metadata (e.g., titles, notes, captions). The
  prompt template resides in `amaf/prompts/contextron.txt`.
- **Visura**: Converts visual cues to semantic tags.  It now also accepts an
  `image_path` and performs OCR before summarising the visual.
  The prompt template resides in `amaf/prompts/visura.txt`.
- **TrendAnalyser**: Spots the two most important numeric trends.
  The prompt template resides in `amaf/prompts/trend_analyser.txt`.
- **SummaCraft**: Performs personalized summarization using CoT prompting. The
  prompt template resides in `amaf/prompts/summa_craft.txt`.







## 📊 Run and Evaluation
We provide a lightweight utility that converts a raw AMAF run folder into a
single JSON file that pairs each **question-id** with its gold answer and your
model’s prediction.  This is the canonical input for scorers that will follow.

### Step 1 – Generate predictions

##TATQA
```
bash run_tatqa.sh --provider lambda --model llama3.1-8b-instruct --method ee_fs --comment myexperiment
```

##FETQA
```
 bash run_fetaqa.sh --provider lambda --model llama3.1-8b-instruct --method ee_fs --comment myexperiment
```



### Step 2 – Consolidate a run
```bash
# example for a TatQA run folder
python ADAF/ADAF/evaluation/scripts/consolidate_run.py \
       --dataset tatqa \
       --run_dir ADAF/ADAF/out/tatqa_logs/llama3.1-8b-instruct_lambda_ee_fs_myexperiment_20250722_145826
```
The script
1. loads the official *dev* gold answers from `ADAF/ADAF/data/<dataset>/…`,
2. parses every `*_out.txt`,
3. creates `ADAF/ADAF/evaluation/<dataset>/<run_folder>/consolidated_<timestamp>.json`.

Each record in that JSON looks like:
```jsonc
{
  "qid": "9337c3e6-c53f-45a9-836a-02c474ceac16",
  "gold_answer": "4.6",
  "gold_scale" : "percent",
  "pred_answer": "4.3",
  "pred_scale" : "percent"
}
```
If a gold or prediction is missing, the corresponding field is `null`.

### Supported datasets
* **tatqa**  – `tatqa_dataset_dev.json`
* **finqa**  – `finqa_dev.json`
* **fetaqa** – `fetaQA-v1_dev.jsonl`
* **mmqa**   – `MMQA_dev.jsonl`

Run `python consolidate_run.py --help` for full CLI options. 

## 🏁 End-to-End Evaluation Pipeline
The `evaluation/` package contains three tiny CLIs that transform raw
AMAF outputs into scored metrics.

Folder anatomy created during a run (example for TatQA):
```
ADAF/ADAF/out/tatqa_logs/<run_folder>/           # raw *_out.txt files
ADAF/ADAF/evaluation/tatqa/<run_folder>/
    consolidated_<ts>.json          # step ①
    postprocessed_<ts>.json         # step ②
    scored_<ts>.json                # step ③ (per-question metrics)
    avg_scores_<ts>.json            # step ③ (dataset averages)
```

Step-by-step
1. **Consolidate** raw logs with gold answers
   ```bash
   python evaluation/scripts/consolidate_run.py \
          --dataset tatqa \
          --run_dir out/tatqa_logs/<run_folder>
   ```
2. **Post-process** (unit conversion, numeric cleaning, etc.)
   ```bash
   python evaluation/scripts/postprocess_run.py \
          --dataset tatqa \
          --consolidated_file evaluation/tatqa/<run_folder>/consolidated_<ts>.json
   ```
3. **Score** to obtain Exact-Match / F1 and summary tables
   ```bash
   python evaluation/scripts/score_run.py \
          --post_file evaluation/tatqa/<run_folder>/postprocessed_<ts>.json
   ```

You can run the three commands in sequence for **finqa**, **fetaqa**, and
**mmqa** by changing the `--dataset` flag and pointing to the correct
directories. 

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

## 📓 Usage
```bash
python agents/summa_craft.py --input data/sample_table.csv --profile "retail investor"
```

## 🧪 Fine-Tuning
Use `notebooks/5_Finetune_Summarizer_Finance.ipynb` to fine-tune a FLAN-T5 or Mistral model using LoRA on financial summarization data.

This repository contains a modular Python implementation of the AMAF pipeline
for personalized, context‑aware table summarization using OpenAI's chat models
(defaults to `gpt-3.5-turbo`; override with the `AMAF_MODEL` environment variable).


## 📊 Evaluation
To assess summarization quality on benchmark datasets we provide `utils/rouge_bleu_eval.py`.
For TAT-QA it converts every ground-truth question/answer pair into a statement and
computes ROUGE-L and BLEU between that concatenated reference and the generated
summary.  FinQA QA accuracy is calculated as the fraction of answers that appear
verbatim (or numerically close) in the summary.  Example usage:

```bash
python utils/rouge_bleu_eval.py --tatqa data/TATQA/tatqa_dataset_dev.json \
    --tatqa_logs out/tatqa_logs \
    --finqa data/sample_finqa.json --finqa_logs out/finqa_logs
```

`AMAF w/o Visura` can be evaluated by omitting visual cues when generating the
summaries to measure the impact of the `Visura` agent.

This repository contains a modular Python implementation of the AMAF pipeline
for personalized, context‑aware table summarization using OpenAI's chat models
(defaults to `gpt-3.5-turbo`; override with the `AMAF_MODEL` environment variable).


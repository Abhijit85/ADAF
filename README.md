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
git clone https://github.com/Abhijit85/amaf-summarization.git
cd amaf-summarization
pip install -r requirements.txt
```

## 🧠 Agent Descriptions
- **TabuSynth**: Extracts structured facts from tabular data using TAPAS or T5
- **Contextron**: Extracts contextual metadata (e.g., titles, notes, captions)
- **Visura**: Converts visual cues (e.g., color/bold/highlight) to semantic tags
- **SummaCraft**: Performs personalized summarization using CoT prompting

## 📓 Usage
```bash
python agents/summa_craft.py --input data/sample_table.csv --profile "retail investor"
```

## 🧪 Fine-Tuning
Use `notebooks/5_Finetune_Summarizer_Finance.ipynb` to fine-tune a FLAN-T5 or Mistral model using LoRA on financial summarization data.

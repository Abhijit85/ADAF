# AMAF: Adaptive Multi-Agent Framework for Table Summarization

AMAF (Adaptive Multi-Agent Framework) is a modular pipeline for personalized and context-aware table summarization using large language models.



## Pipeline Diagram
![Alt text](Pipeline-ADAF.png)




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

# NLP Medical Data Extraction

## Project Overview

This project focuses on fine-tuning a transformer-based model for named-entity recognition (token classification) on radiology and medical text reports. The goal is to automatically extract key entities (e.g., findings, impressions, clinical history) from unstructured clinical narratives, enabling faster data processing and analysis.

Key features:

* Fine-tuned a pretrained BERT model for token classification.
* Evaluated model performance on held-out medical reports.
* Integrated Metaflow for scalable, reproducible workflows.
* Deployed a Streamlit app for interactive model demos.

## Evaluation Results

The following metrics were achieved on the evaluation dataset:

```
eval_loss: 0.052343737334012985
eval_precision: 0.987212377109425
eval_recall:    0.9871664733178654
eval_f1:        0.9871175118005348
eval_runtime:   54.4215 s
eval_samples_per_second: 3.51
```

These results demonstrate high precision and recall, indicating the model reliably identifies relevant entities in radiology reports.

## Requirements

* Python 3.8+
* Git
* [Conda](https://docs.conda.io/) (recommended) or virtualenv

Python dependencies (install via `requirements.txt`):

```
transformers
datasets
torch
scikit-learn
seqeval
numpy
pandas
metaflow
streamlit
```

## Setup Instructions

1. **Clone the repository**

   ```bash
   git clone https://github.com/aakashsgowda/BuildProjects_OpenAvenue.git
   cd BuildProjects_OpenAvenue
   ```
2. **Create and activate a Python environment**

   ```bash
   conda create -n nlp_med python=3.8 -y
   conda activate nlp_med
   # or using virtualenv
   # python -m venv venv
   # source venv/bin/activate
   ```
3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```
4. **Prepare data**

   * Place your raw medical reports in `data/raw/`.
   * Run the preprocessing script:

     ```bash
     python src/preprocess.py --input_dir data/raw --output_dir data/processed
     ```

## Metaflow Workflow

We use Metaflow to define and run reproducible pipelines. To execute the end-to-end flow:

```bash
python src/flow.py run --environment=conda
```

This covers data preprocessing, training, and evaluation steps automatically.

## Streamlit App

An interactive demo allows you to input clinical text and visualize extracted entities.

1. **Launch the app**

   ```bash
   streamlit run app/streamlit_app.py
   ```
2. Open the provided URL (usually [http://localhost:8501/](http://localhost:8501/)) in your browser.
3. Enter text in the sidebar and view real-time entity highlights.

## Training

To fine-tune the model separately, run:

```bash
python src/train.py \
  --train_data data/processed/train.json \
  --eval_data  data/processed/eval.json \
  --output_dir models/clinical-token-classifier \
  --epochs 3 \
  --batch_size 16 \
  --learning_rate 5e-5
```

Adjust hyperparameters as needed.

## Evaluation

After training, evaluate your model:

```bash
python src/evaluate.py \
  --model_dir models/clinical-token-classifier \
  --eval_data data/processed/eval.json
```

## Usage

Load the trained model in your application:

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification
from src.evaluate import predict_entities

# Load
tokenizer = AutoTokenizer.from_pretrained("models/clinical-token-classifier")
model = AutoModelForTokenClassification.from_pretrained("models/clinical-token-classifier")

# Predict
text = "CHEST RADIOGRAPHY EXAM DATE: ..."
entities = predict_entities(text, tokenizer, model)
print(entities)
```

## License

This project is licensed under the MIT License. Feel free to use and modify as needed.

# PetroText AI — Petroleum Engineering Text Classification

## Project Overview

PetroText AI is a Natural Language Processing (NLP) project that classifies petroleum engineering research abstracts into five technical categories using traditional machine learning and Transformer-based deep learning.

The project compares a TF-IDF + Logistic Regression baseline with a fine-tuned DistilBERT model.

## Classification Categories

1. Reservoir Engineering
2. Production Engineering
3. Drilling Engineering
4. Well Integrity
5. Geology & Petrophysics

## Dataset

- Source: OpenAlex research paper metadata and abstracts
- Total abstracts: 158
- Training samples: 110
- Validation samples: 24
- Test samples: 24
- Number of categories: 5

The dataset was filtered and manually reviewed for topic relevance. The dataset is relatively small and class-imbalanced, which limits generalization.

## Methodology

### Model 1: TF-IDF + Logistic Regression

- Convert research abstracts into numerical TF-IDF features.
- Train a multiclass Logistic Regression classifier.
- Evaluate classification performance.

### Model 2: DistilBERT

- Tokenize abstracts using the pretrained DistilBERT tokenizer.
- Apply padding and truncation to a maximum of 256 tokens.
- Fine-tune DistilBERT for five epochs using PyTorch.
- Use AdamW with a learning rate of 2e-5.
- Predict one of five engineering categories.

## Results

| Metric | Logistic Regression | DistilBERT |
|---|---:|---:|
| Test Accuracy | 87.50% | 87.50% |
| Test Macro F1 | 0.8629 | 0.8910 |

Both models correctly classified 21 of the 24 held-out test abstracts.

DistilBERT achieved a slightly higher macro F1, while Logistic Regression provided comparable accuracy with substantially lower computational cost.

Because the test dataset contains only 24 examples, the results should be interpreted as preliminary.

## Technologies

Python, pandas, NumPy, scikit-learn, PyTorch, Hugging Face Transformers, and joblib.

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the interactive classifier:

```bash
python predict.py
```

Enter a petroleum engineering abstract to receive a predicted category.

The script requires the fine-tuned model and tokenizer files in `models/distilbert/`.

## Limitations and Future Improvements

- Expand and improve the labeled dataset.
- Investigate class imbalance and ambiguous category boundaries.
- Evaluate on a larger independent test dataset.
- Improve performance on short technical sentences.
- Add confidence scores and a web interface.

## Data and Reproducibility

Abstracts were retrieved through OpenAlex. Abstract redistribution rights have not been independently verified, so the collected CSV files are excluded from the public repository.

Reproducing the reported results requires access to the underlying dataset and the saved model artifacts.

## Project Status

Research prototype demonstrating traditional NLP classification and Transformer fine-tuning for petroleum engineering text.
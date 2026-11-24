
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load saved tokenizer
tokenizer = AutoTokenizer.from_pretrained("models/distilbert")

# Load fine-tuned DistilBERT
model = AutoModelForSequenceClassification.from_pretrained(
    "models/distilbert"
)

model.eval()

print("PetroText AI model loaded successfully!")

import torch

# Our five engineering categories
categories = [
    "Reservoir Engineering",
    "Production Engineering",
    "Drilling Engineering",
    "Well Integrity",
    "Geology & Petrophysics"
]

# New sentence
text = input("Enter a petroleum engineering abstract: ")

# Tokenize input
inputs = tokenizer(
    text,
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=256
)

# Predict without training
with torch.no_grad():
    outputs = model(**inputs)

# Select category with highest score
predicted_label = torch.argmax(outputs.logits, dim=1).item()

print("Input:", text)
print("Predicted category:", categories[predicted_label])

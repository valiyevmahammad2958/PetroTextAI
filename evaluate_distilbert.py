
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score

# Load test data
test = pd.read_csv("data/test.csv")

# Load saved fine-tuned model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("models/distilbert")

model = AutoModelForSequenceClassification.from_pretrained(
    "models/distilbert"
)

# Tokenize test abstracts
test_encodings = tokenizer(
    test["abstract"].tolist(),
    padding="max_length",
    truncation=True,
    max_length=256,
    return_tensors="pt"
)

# Switch model to evaluation mode
model.eval()

# Make predictions
with torch.no_grad():
    outputs = model(
        input_ids=test_encodings["input_ids"],
        attention_mask=test_encodings["attention_mask"]
    )

# Convert model scores into predicted categories
predictions = torch.argmax(
    outputs.logits,
    dim=1
).numpy()

# Calculate test metrics
accuracy = accuracy_score(test["label"], predictions)

macro_f1 = f1_score(
    test["label"],
    predictions,
    average="macro",
    zero_division=0
)

print("Test samples:", len(test))
print("DistilBERT test accuracy:", round(accuracy, 4))
print("DistilBERT test macro F1:", round(macro_f1, 4))
print("Correct predictions:", sum(predictions == test["label"]))


category_names = [
    "Reservoir Engineering",
    "Production Engineering",
    "Drilling Engineering",
    "Well Integrity",
    "Geology & Petrophysics"
]

print("\nMisclassified test abstracts:")

for i in range(len(test)):
    actual = int(test["label"].iloc[i])
    predicted = int(predictions[i])

    if actual != predicted:
        print("\nAbstract:", i + 1)
        print("Actual:", category_names[actual])
        print("Predicted:", category_names[predicted])
        print("Text:", test["abstract"].iloc[i][:300])

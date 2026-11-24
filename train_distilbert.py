
from transformers import AutoTokenizer

# Load pretrained DistilBERT tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "distilbert-base-uncased"
)

# Example petroleum engineering sentence
text = "Reservoir permeability affects oil production."

# Convert sentence into tokens
tokens = tokenizer.tokenize(text)

print("Original text:", text)
print("Tokens:", tokens)
print("Number of tokens:", len(tokens))

# Convert the sentence into numerical token IDs
encoded = tokenizer(text)

print("Input IDs:", encoded["input_ids"])
print("Attention mask:", encoded["attention_mask"])

# Convert IDs back into readable tokens
print("Tokens:", tokenizer.convert_ids_to_tokens(
    encoded["input_ids"]
))


# Tokenize with a fixed sequence length
encoded_fixed = tokenizer(
    text,
    padding="max_length",
    truncation=True,
    max_length=16
)

print("Input IDs:", encoded_fixed["input_ids"])
print("Attention mask:", encoded_fixed["attention_mask"])
print("Tokens:", tokenizer.convert_ids_to_tokens(
    encoded_fixed["input_ids"]
))

import pandas as pd

# Load training dataset
train = pd.read_csv("data/train.csv")

# Extract abstracts and labels
X_train = train["abstract"].tolist()
y_train = train["label"].tolist()

# Tokenize all training abstracts
train_encodings = tokenizer(
    X_train,
    padding="max_length",
    truncation=True,
    max_length=256,
    return_tensors="pt"
)

print("Training abstracts:", len(X_train))
print("Input IDs shape:", train_encodings["input_ids"].shape)
print("Attention mask shape:", train_encodings["attention_mask"].shape)
print("Training labels:", len(y_train))


import torch
from torch.utils.data import TensorDataset

# Convert labels to PyTorch tensor
labels_tensor = torch.tensor(y_train, dtype=torch.long)

# Combine inputs, masks, and labels
train_dataset = TensorDataset(
    train_encodings["input_ids"],
    train_encodings["attention_mask"],
    labels_tensor
)

print("Dataset size:", len(train_dataset))

# Inspect first training example
input_ids, attention_mask, label = train_dataset[0]

print("First input shape:", input_ids.shape)
print("First mask shape:", attention_mask.shape)
print("First label:", label.item())


from torch.utils.data import DataLoader

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True
)

print("Number of batches:", len(train_loader))

# Inspect the first batch
batch = next(iter(train_loader))

input_ids, attention_mask, labels = batch

print("Input shape:", input_ids.shape)
print("Attention mask shape:", attention_mask.shape)
print("Labels shape:", labels.shape)



from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=5
)

print("DistilBERT model loaded!")
print("Number of output categories:", model.config.num_labels)
print("Model type:", model.config.model_type)


from torch.optim import AdamW

# Choose CPU or GPU
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Move model to selected device
model.to(device)

# Create optimizer
optimizer = AdamW(
    model.parameters(),
    lr=2e-5
)

print("Training device:", device)
print("Optimizer:", type(optimizer).__name__)
print("Learning rate:", optimizer.param_groups[0]["lr"])


# Set model to training mode
model.train()

total_loss = 0

# Train for one epoch

num_epochs = 5

for epoch in range(num_epochs):

    model.train()
    total_loss = 0

    for batch in train_loader:

        input_ids, attention_mask, labels = batch

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)

    print(
        f"Epoch {epoch + 1}/{num_epochs} - "
        f"Average loss: {average_loss:.4f}"
    )

from sklearn.metrics import accuracy_score, f1_score

# Load validation data
val = pd.read_csv("data/val.csv")

# Tokenize validation abstracts
val_encodings = tokenizer(
    val["abstract"].tolist(),
    padding="max_length",
    truncation=True,
    max_length=256,
    return_tensors="pt"
)

# Switch to evaluation mode
model.eval()

# Make predictions without updating weights
with torch.no_grad():
    outputs = model(
        input_ids=val_encodings["input_ids"].to(device),
        attention_mask=val_encodings["attention_mask"].to(device)
    )

# Convert category scores to predicted labels
predictions = torch.argmax(
    outputs.logits, dim=1
).cpu().numpy()

# Calculate metrics
accuracy = accuracy_score(val["label"], predictions)
macro_f1 = f1_score(
    val["label"],
    predictions,
    average="macro",
    zero_division=0
)

print("Validation accuracy:", round(accuracy, 4))
print("Validation macro F1:", round(macro_f1, 4))
print("Correct predictions:", sum(predictions == val["label"]))


# Save fine-tuned DistilBERT and tokenizer
model.save_pretrained("models/distilbert")
tokenizer.save_pretrained("models/distilbert")

print("Fine-tuned DistilBERT saved successfully!")

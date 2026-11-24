
import pandas as pd
from sklearn.model_selection import train_test_split

# Load latest cleaned dataset
df = pd.read_csv("data/openalex_abstracts_relevant_v5.csv")

# First split: 70% train, 30% temporary
train, temp = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    stratify=df["label"]
)

# Second split: divide temporary data equally
val, test = train_test_split(
    temp,
    test_size=0.50,
    random_state=42,
    stratify=temp["label"]
)

# Save new splits
train.to_csv("data/train.csv", index=False)
val.to_csv("data/val.csv", index=False)
test.to_csv("data/test.csv", index=False)

print("Training samples:", len(train))
print("Validation samples:", len(val))
print("Test samples:", len(test))

print("\nTraining category distribution:")
print(train["label"].value_counts().sort_index())

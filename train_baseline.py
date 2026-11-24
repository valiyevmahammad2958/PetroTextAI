
import pandas as pd

# Load training data
train = pd.read_csv("data/train.csv")

# Inputs: research abstracts
X_train = train["abstract"]

# Targets: engineering categories
y_train = train["label"]

print("Training samples:", len(X_train))
print("First abstract:", X_train.iloc[0][:200])
print("First label:", y_train.iloc[0])


from sklearn.feature_extraction.text import TfidfVectorizer

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words="english")

# Learn vocabulary and convert abstracts to numbers
X_train_tfidf = vectorizer.fit_transform(X_train)

print("TF-IDF matrix shape:", X_train_tfidf.shape)
print("Vocabulary size:", len(vectorizer.vocabulary_))



from sklearn.linear_model import LogisticRegression

# Create the classification model
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train_tfidf, y_train)

print("Logistic Regression training complete!")
print("Classes learned:", model.classes_)
print("Coefficient matrix shape:", model.coef_.shape)



from sklearn.metrics import accuracy_score

# Load validation data
val = pd.read_csv("data/val.csv")

X_val = val["abstract"]
y_val = val["label"]

# Convert validation abstracts to TF-IDF
X_val_tfidf = vectorizer.transform(X_val)

# Predict engineering categories
y_pred = model.predict(X_val_tfidf)

# Calculate accuracy
accuracy = accuracy_score(y_val, y_pred)

print("Validation samples:", len(X_val))
print("Validation accuracy:", round(accuracy, 3))
print("Correct predictions:", sum(y_pred == y_val))


from sklearn.metrics import classification_report

category_names = [
    "Reservoir Engineering",
    "Production Engineering",
    "Drilling Engineering",
    "Well Integrity",
    "Geology & Petrophysics"
]

print("\nClassification Report:")
print(
    classification_report(
        y_val,
        y_pred,
        labels=[0, 1, 2, 3, 4],
        target_names=category_names,
        zero_division=0
    )
)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(
    y_val,
    y_pred,
    labels=[0, 1, 2, 3, 4]
)

print("\nConfusion Matrix:")
print(cm)


print("\nMisclassified Abstracts:")

for i in range(len(y_val)):
    if y_val.iloc[i] != y_pred[i]:
        print("\n----------------------")
        print("Actual:", category_names[y_val.iloc[i]])
        print("Predicted:", category_names[y_pred[i]])
        print("Abstract:", X_val.iloc[i][:500])

import joblib
import os

# Create folder for saved models
os.makedirs("models", exist_ok=True)

# Save trained Logistic Regression model
joblib.dump(model, "models/logistic_regression.joblib")

# Save fitted TF-IDF vectorizer
joblib.dump(vectorizer, "models/tfidf_vectorizer.joblib")

print("Baseline model and vectorizer saved!")

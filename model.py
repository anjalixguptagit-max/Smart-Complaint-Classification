import pandas as pd
import json
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("data/complaints_processed.csv")


# ============================================================
# 2. DATA CLEANING
# ============================================================

# Remove unnecessary index column
df = df.drop(columns=["Unnamed: 0"])

# Remove rows where complaint text is missing
df = df.dropna(subset=["narrative"])


# ============================================================
# 3. INPUT AND TARGET
# ============================================================

X = df["narrative"]
y = df["product"]


# ============================================================
# 4. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)



train_data = pd.DataFrame({
    "narrative": X_train,
    "product": y_train
})

train_data.to_csv("data/training_data.csv", index=False)


print("Total data:", len(df))
print("Training data:", len(X_train))
print("Testing data:", len(X_test))


# ============================================================
# 5. TF-IDF TEXT VECTORIZATION
# ============================================================

tfidf = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2)
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)


# ============================================================
# 6. TRAIN LINEAR SVM MODEL
# ============================================================

model = LinearSVC()

model.fit(X_train_tfidf, y_train)


# ============================================================
# 7. PREDICTION
# ============================================================

y_pred = model.predict(X_test_tfidf)


# ============================================================
# 8. MODEL EVALUATION
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

report = classification_report(
    y_test,
    y_pred,
    output_dict=True
)


print("\nModel Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ============================================================
# 9. SAVE TRAINED MODEL
# ============================================================

joblib.dump(model, "model.pkl")
joblib.dump(tfidf, "tfidf.pkl")


# ============================================================
# 10. SAVE PERFORMANCE METRICS
# ============================================================

metrics = {
    "accuracy": accuracy,
    "macro_f1": report["macro avg"]["f1-score"],
    "weighted_f1": report["weighted avg"]["f1-score"],
    "training_samples": len(X_train),
    "testing_samples": len(X_test),
    "categories": len(model.classes_)
}


with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)


# ============================================================
# 11. FINAL STATUS
# ============================================================

print("\nModel and TF-IDF vectorizer saved successfully!")
print("Performance metrics saved successfully!")

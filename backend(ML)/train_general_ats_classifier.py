import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# 1) Load dataset
df = pd.read_csv("Resume.csv")  # from your ZIP

# Use plain text column
texts = df["Resume_str"].fillna("")

# 2) Simple cleaning
def clean_text(t):
    t = re.sub(r"\s+", " ", str(t))
    return t.lower()

texts = texts.apply(clean_text)

# 3) Auto-labeling (proxy labels)
# Heuristics to create Poor / Average / Good labels
def auto_label(text):
    words = text.split()
    length = len(words)

    sections = 0
    for s in ["skills", "experience", "education", "projects"]:
        if s in text:
            sections += 1

    if length < 200 or sections <= 1:
        return "Poor"
    elif length < 500 or sections <= 2:
        return "Average"
    else:
        return "Good"

labels = texts.apply(auto_label)

# 4) Vectorize text
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X = vectorizer.fit_transform(texts)
y = labels

# 5) Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6) Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 7) Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# 8) Save model + vectorizer
joblib.dump(model, "general_ats_classifier.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("✅ Model saved: general_ats_classifier.pkl")
print("✅ Vectorizer saved: tfidf_vectorizer.pkl")
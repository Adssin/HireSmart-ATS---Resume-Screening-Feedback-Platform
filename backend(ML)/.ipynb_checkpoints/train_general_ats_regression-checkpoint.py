import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# 1) Load data
train_df = pd.read_csv("train.csv")
val_df = pd.read_csv("validation.csv")

# 2) Basic cleaning
def clean_text(t):
    t = str(t)
    t = re.sub(r"\s+", " ", t)
    return t.lower().strip()

X_train_text = train_df["text"].apply(clean_text)
y_train = train_df["ats_score"]

X_val_text = val_df["text"].apply(clean_text)
y_val = val_df["ats_score"]

# 3) Vectorize text
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000,
    ngram_range=(1, 2)
)

X_train = vectorizer.fit_transform(X_train_text)
X_val = vectorizer.transform(X_val_text)

# 4) Train regression model
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

# 5) Evaluate
y_pred = model.predict(X_val)
mae = mean_absolute_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

print("Validation MAE:", mae)
print("Validation R2:", r2)

# 6) Save model and vectorizer
joblib.dump(model, "general_ats_regressor.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("✅ Saved: general_ats_regressor.pkl")
print("✅ Saved: tfidf_vectorizer.pkl")
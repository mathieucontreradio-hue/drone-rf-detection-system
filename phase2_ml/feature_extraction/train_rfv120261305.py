import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# load dataset
df = pd.read_csv("dataset_rf.csv")

X = df.drop(columns=["label"])
y = df["label"]

# split dataset (stratify = classes équilibrées)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# model RF (robuste bruit SDR)
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    n_jobs=-1,
    random_state=42
)

model.fit(X_train, y_train)

# evaluation
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))

# save model
joblib.dump(model, "rf_model.pkl")

print("Model saved ✔")

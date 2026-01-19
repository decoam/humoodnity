import pandas as pd

df = pd.read_csv("mood_dataset.csv")

print(df.head())
print(df.info())

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

X = df[["tidur_jam", "stres", "capek", "aktivitas"]]
y = df["mood"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["aktivitas"]),
        ("num", "passthrough", ["tidur_jam", "stres", "capek"]),
    ]
)

model = LogisticRegression(max_iter=2000)

clf = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", model)
])

clf.fit(X_train, y_train)

contoh = {
    "tidur_jam": 7,
    "stres": 3,
    "capek": 2,
    "aktivitas": "kerja"
}

import pandas as pd
contoh_df = pd.DataFrame([contoh])

pred = clf.predict(contoh_df)[0]
print("Prediksi mood:", pred)


import joblib
joblib.dump(clf, "mood_model.pkl")
print("Model tersimpan ke mood_model.pkl")

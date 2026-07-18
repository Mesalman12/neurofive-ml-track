"""
Model Evaluation & Tuning: Beyond Accuracy
============================================
Titanic survival classification — using the REAL Kaggle Titanic dataset
(891 rows, from your neurofive-ml-track repo's titanic.csv).
"""

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

RANDOM_STATE = 42

# ---------------------------------------------------------------------
# 1. Load real data
# ---------------------------------------------------------------------
df = pd.read_csv("data/titanic.csv")
print("Class balance (Survived):")
print(df["Survived"].value_counts(normalize=True).round(3), "\n")

# ---------------------------------------------------------------------
# 2. Preprocess (same steps as your Task 1/2 EDA + cleaning)
# ---------------------------------------------------------------------
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])

df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

X = df.drop(columns=["Survived"])
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

# ---------------------------------------------------------------------
# 3. BASELINE model (defaults, no tuning) -- this is "your previous task"
# ---------------------------------------------------------------------
baseline = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(random_state=RANDOM_STATE)),
    ]
)
baseline.fit(X_train, y_train)
y_pred_base = baseline.predict(X_test)

print("=" * 60)
print("BASELINE MODEL (default hyperparameters)")
print("=" * 60)
print(classification_report(y_test, y_pred_base, target_names=["Died", "Survived"]))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_base))

base_metrics = {
    "accuracy": accuracy_score(y_test, y_pred_base),
    "precision": precision_score(y_test, y_pred_base),
    "recall": recall_score(y_test, y_pred_base),
    "f1": f1_score(y_test, y_pred_base),
}

# ---------------------------------------------------------------------
# 4. GridSearchCV -- tune n_estimators, max_depth, min_samples_split
# ---------------------------------------------------------------------
param_grid = {
    "clf__n_estimators": [100, 200, 400],
    "clf__max_depth": [None, 4, 8, 12],
    "clf__min_samples_split": [2, 5, 10],
}

grid = GridSearchCV(
    estimator=Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(random_state=RANDOM_STATE)),
        ]
    ),
    param_grid=param_grid,
    scoring="f1",
    cv=5,
    n_jobs=-1,
)
grid.fit(X_train, y_train)

print("\nBest params found by GridSearchCV:", grid.best_params_)
print("Best CV F1 score:", round(grid.best_score_, 4))

tuned_model = grid.best_estimator_
y_pred_tuned = tuned_model.predict(X_test)

print("\n" + "=" * 60)
print("TUNED MODEL (GridSearchCV, scoring='f1')")
print("=" * 60)
print(classification_report(y_test, y_pred_tuned, target_names=["Died", "Survived"]))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_tuned))

tuned_metrics = {
    "accuracy": accuracy_score(y_test, y_pred_tuned),
    "precision": precision_score(y_test, y_pred_tuned),
    "recall": recall_score(y_test, y_pred_tuned),
    "f1": f1_score(y_test, y_pred_tuned),
}

# ---------------------------------------------------------------------
# 5. Before / After comparison table
# ---------------------------------------------------------------------
comparison = pd.DataFrame(
    {
        "Baseline (default)": base_metrics,
        "Tuned (GridSearchCV)": tuned_metrics,
    }
).T
comparison["improvement (F1)"] = comparison["f1"] - base_metrics["f1"]
comparison = comparison.round(4)

print("\n" + "=" * 60)
print("BEFORE vs AFTER")
print("=" * 60)
print(comparison.to_string())

comparison.to_csv("before_after_comparison_real.csv")

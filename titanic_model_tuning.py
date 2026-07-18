"""
Model Evaluation & Tuning: Beyond Accuracy
============================================
Titanic (survival) classification task.

NOTE ON DATA: This sandbox has no internet access, so the classic
train.csv from Kaggle/seaborn can't be downloaded here. This script
builds a synthetic-but-realistic Titanic-style dataset (same columns,
same kind of class imbalance, same feature relationships) so every
step runs end-to-end. If you already have your own Titanic train.csv
from the previous task, just replace the `build_dataset()` call with:

    df = pd.read_csv("train.csv")

everything else (metrics, tuning, comparison table) works unchanged.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
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
rng = np.random.default_rng(RANDOM_STATE)


# ---------------------------------------------------------------------
# 1. Build a Titanic-style dataset (imbalanced: ~62% died / 38% survived)
# ---------------------------------------------------------------------
def build_dataset(n=891):
    pclass = rng.choice([1, 2, 3], size=n, p=[0.24, 0.21, 0.55])
    sex = rng.choice(["male", "female"], size=n, p=[0.65, 0.35])
    age = np.clip(rng.normal(29, 13, size=n), 0.5, 80)
    sibsp = rng.poisson(0.5, size=n)
    parch = rng.poisson(0.4, size=n)
    fare = np.clip(
        rng.gamma(2.0, scale=(40 - (pclass - 1) * 12), size=n), 4, 512
    )
    embarked = rng.choice(["S", "C", "Q"], size=n, p=[0.72, 0.19, 0.09])

    # realistic-ish survival probability model (women/children/1st class favored)
    logit = (
        -2.1
        + 1.9 * (sex == "female")
        + 0.9 * (pclass == 1)
        + 0.3 * (pclass == 2)
        + 0.02 * (35 - age)
        + 0.004 * fare
        - 0.15 * sibsp
    )
    prob = 1 / (1 + np.exp(-logit))
    survived = rng.binomial(1, prob)

    df = pd.DataFrame(
        {
            "Pclass": pclass,
            "Sex": sex,
            "Age": age.round(1),
            "SibSp": sibsp,
            "Parch": parch,
            "Fare": fare.round(2),
            "Embarked": embarked,
            "Survived": survived,
        }
    )
    return df


df = build_dataset()
print("Class balance (Survived):")
print(df["Survived"].value_counts(normalize=True).round(3), "\n")

# ---------------------------------------------------------------------
# 2. Preprocess
# ---------------------------------------------------------------------
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
    scoring="f1",          # optimize for F1, not accuracy -- see write-up
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

comparison.to_csv("/home/claude/before_after_comparison.csv")

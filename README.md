# Task 3: Model Evaluation & Tuning — Beyond Accuracy

### Why accuracy alone is misleading here
Survival on the Titanic is imbalanced — 61.6% died vs. 38.4% survived
in this dataset. A model that predicted "died" for every single
passenger would score **~62% accuracy** while being completely
useless — it would never correctly identify a single survivor.
Accuracy treats both classes as equally important, so it can't tell
the difference between a genuinely good model and a lazy one that
just leans on the majority class. **Precision, recall, and F1** fix
this by scoring each class separately:

- **Precision** — of everyone the model predicted "Survived," how
  many actually survived? (cost of false alarms)
- **Recall** — of everyone who actually survived, how many did the
  model catch? (cost of missed cases)
- **F1** — the harmonic mean of the two, so a model can't hide behind
  being good at only one of them.

### Before vs. After
`GridSearchCV` (5-fold CV, scored on F1) tuned `n_estimators`,
`max_depth`, and `min_samples_split` on a `RandomForestClassifier`.

| Metric | Baseline (default RF) | Tuned (GridSearchCV) | Change |
|---|---|---|---|
| Accuracy | 0.8045 | 0.7989 | -0.0056 |
| Precision | 0.7833 | 0.7797 | -0.0036 |
| Recall | 0.6812 | 0.6667 | -0.0145 |
| F1 | 0.7287 | 0.7188 | -0.0099 |

Best params found: `max_depth=None`, `min_samples_split=10`,
`n_estimators=200`.

### What tuning actually changed
On this run, tuning **didn't improve** the model — it came in very
slightly behind the untuned default `RandomForestClassifier` on every
metric, including F1. A few honest takeaways from that:

1. `GridSearchCV`'s reported best CV score (0.7656) was measured
   across 5 folds of the *training* data; the held-out test set can
   (and did) come in a bit lower. That gap is expected and is exactly
   why you always confirm on a separate test set instead of trusting
   the CV score alone.
2. Random Forest's own default settings (100 trees, unlimited depth,
   min_samples_split=2) are already a fairly strong baseline —
   scikit-learn's defaults are tuned to be reasonable out of the box,
   so there isn't always a lot of room to improve on a small (~900
   row), already-somewhat-clean dataset like Titanic.
3. This is a useful, real lesson: hyperparameter tuning is a search
   for *potential* improvement, not a guarantee of one. Comparing
   before/after on F1 (not just accuracy) is what caught this —
   accuracy alone barely moved (0.8045 → 0.7989) and wouldn't have
   raised a flag either way.

### Files
- `titanic_model_tuning_real.py` — full pipeline on the real
  `titanic.csv`: preprocessing, baseline model, `classification_report`,
  `GridSearchCV`, before/after comparison.
- `before_after_comparison_real.csv` — raw metrics table.

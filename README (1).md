# neurofive-ml-track

## Task 1: Data Science Toolkit Setup & First EDA

This repo tracks my progress through the Neurofive ML track.

### What's here
- `titanic_eda.ipynb` — first exploratory data analysis on the Titanic dataset (891 rows, 12 columns). Covers `.info()`, `.describe()`, `.head()`, missing-value audit, categorical vs. numerical breakdown, and a short data-story writeup.
- `titanic.csv` — the dataset used (Kaggle: "Titanic - Machine Learning from Disaster").

### Environment
- Python 3
- pandas, numpy, jupyter (or run directly in Google Colab)

### Key findings (Task 1 — EDA)
- 891 rows, 12 columns
- Missing values: `Age` (177, ~20%), `Cabin` (687, ~77%), `Embarked` (2)
- Numerical: PassengerId, Survived, Pclass, Age, SibSp, Parch, Fare
- Categorical: Name, Sex, Ticket, Cabin, Embarked

## Task 2: Clean & Visualize Real-World Data

- **Age** → filled with median (robust to right skew)
- **Embarked** → filled with mode ('S', only 2 rows affected)
- **Cabin** → converted to a binary `HasCabin` flag instead of imputing (77% missing was too sparse)
- **Outlier check**: boxplot on `Fare` — high fares are legitimate 1st-class tickets, not errors, so kept in
- **4 visualizations**: histogram (Age), boxplot (Fare by Pclass), bar chart (survival rate by Sex), correlation heatmap
- **Answer — feature most affecting survival**: `Sex` is the strongest single predictor (~74% survival for women vs ~19% for men), consistent with the "women and children first" evacuation policy. `Pclass`/`Fare` are the next strongest signals.

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# -----------------------------
# Config
# -----------------------------
RANDOM_STATE = 42
CSV_PATH = "/opt/notebooks/FinalFeatures/ICA_Pearson_Full_Features.csv"
TARGET_COL = "ptsd"
DROP_DEMOG_COLS = 8
PERCENTILE = 50

C_GRID = [0.01, 0.1, 1.0]
L1_RATIO_GRID = [0.1, 0.3, 0.5, 0.7, 0.9]

OUTER = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
INNER = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# Data 
df = pd.read_csv(CSV_PATH).dropna()
df = df.iloc[:, 1:]  # index column

y = df[TARGET_COL].astype(int).to_numpy()
X = df.drop(columns=[TARGET_COL]).iloc[:, :-DROP_DEMOG_COLS].to_numpy(np.float32)


pipe = Pipeline(
    steps=[
        ("select", SelectPercentile(f_classif, percentile=PERCENTILE)),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            max_iter=5000,
            tol=1e-3,
            random_state=RANDOM_STATE,
        )),
    ]
)

param_grid = {"clf__C": C_GRID, "clf__l1_ratio": L1_RATIO_GRID}

# hyperparams by inner-CV 
search = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring="balanced_accuracy",
    cv=INNER,
    refit=True,
    n_jobs=-1,
)

def fold_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = recall_score(y_true, y_pred, zero_division=0)            # = TP/(TP+FN)
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": sens,
        "sensitivity": sens,
        "specificity": spec,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


# Nested CV 
inner_rows, outer_rows = [], []

for outer_fold, (tr, te) in enumerate(OUTER.split(X, y)):
    search.fit(X[tr], y[tr])          
    best = search.best_params_

    cv = pd.DataFrame(search.cv_results_)
    inner_rows.append(
        cv.assign(
            outer_fold=outer_fold,
            C=cv["param_clf__C"].astype(float),
            l1_ratio=cv["param_clf__l1_ratio"].astype(float),
        )[["outer_fold", "C", "l1_ratio", "mean_test_score", "std_test_score", "rank_test_score"]]
        .rename(columns={
            "mean_test_score": "mean_balanced_accuracy",
            "std_test_score": "std_balanced_accuracy",
            "rank_test_score": "rank",
        })
    )

    y_hat = search.best_estimator_.predict(X[te])
    outer_rows.append({"outer_fold": outer_fold, **best, **fold_metrics(y[te], y_hat)})

inner_df = pd.concat(inner_rows, ignore_index=True)
outer_df = pd.DataFrame(outer_rows)

inner_df.to_csv("inner_metrics.csv", index=False)
outer_df.to_csv("outer_metrics.csv", index=False)
outer_df[["outer_fold", "tn", "fp", "fn", "tp", "sensitivity", "specificity", "balanced_accuracy"]].to_csv(
    "outer_confusion_and_rates.csv", index=False
)

print("DONE")
print(outer_df[["balanced_accuracy", "accuracy", "f1"]].mean())

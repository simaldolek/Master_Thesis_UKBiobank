import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.svm import SVC
from sklearn.metrics import (
    balanced_accuracy_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)

RANDOM_STATE = 42
CSV_PATH = "/opt/notebooks/FinalFeatures/ICA_Partial_Tikhonov_Full_Features.csv"
TARGET_COL = "ptsd"
DROP_DEMOG_COLS = 8

OUTER = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
INNER = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


df = pd.read_csv(CSV_PATH).dropna().iloc[:, 1:]  # drop index-like first col if present
y = df[TARGET_COL].astype(int).to_numpy()

X_df = df.drop(columns=[TARGET_COL]).iloc[:, :-DROP_DEMOG_COLS]
feature_names = X_df.columns.to_numpy()
X = X_df.to_numpy(np.float32)


pipe = Pipeline(
    steps=[
        ("screen", SelectPercentile(score_func=f_classif)),
        ("scale", StandardScaler()),
        # Linear SVM for comparable “linear model” behavior; probability=True enables ROC-AUC
        ("clf", SVC(kernel="linear", probability=True, class_weight="balanced", random_state=RANDOM_STATE)),
    ]
)

param_grid = {
    # also ran 10, 30, 50 - same performance
    "screen__percentile": [50, 70, 80],
    "clf__C": [0.01, 0.1, 1.0],
}

search = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring="balanced_accuracy",
    cv=INNER,
    refit=True,
    n_jobs=-1,
)

def fold_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = recall_score(y_true, y_pred, zero_division=0)
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": sens,
        "sensitivity": sens,
        "specificity": spec,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


# Nested cross validation
inner_rows, outer_rows, topfeat_rows = [], [], []

for outer_fold, (tr, te) in enumerate(OUTER.split(X, y), start=1):
    search.fit(X[tr], y[tr])
    best = search.best_params_
    best_est = search.best_estimator_

    # Inner 
    cv = pd.DataFrame(search.cv_results_)
    inner_rows.append(
        cv.assign(
            outer_fold=outer_fold,
            C=cv["param_clf__C"].astype(float),
            percentile=cv["param_screen__percentile"].astype(float),
        )[
            [
                "outer_fold",
                "percentile",
                "C",
                "mean_test_score",
                "std_test_score",
                "rank_test_score",
            ]
        ].rename(
            columns={
                "mean_test_score": "mean_balanced_accuracy",
                "std_test_score": "std_balanced_accuracy",
                "rank_test_score": "rank",
            }
        )
    )

    # Outer
    y_hat = best_est.predict(X[te])
    y_prob = best_est.predict_proba(X[te])[:, 1]
    auc = roc_auc_score(y[te], y_prob)

    outer_rows.append(
        {
            "outer_fold": outer_fold,
            "auc": float(auc),
            **best,
            **fold_metrics(y[te], y_hat),
        }
    )

    screen = best_est.named_steps["screen"]
    scores = np.nan_to_num(screen.scores_.copy(), nan=-np.inf)
    top_idx = np.argsort(scores)[::-1][:50]

    for rank, j in enumerate(top_idx, start=1):
        topfeat_rows.append(
            {
                "outer_fold": outer_fold,
                "rank": rank,
                "feature": feature_names[j],
                "f_score": float(scores[j]),
            }
        )


inner_df = pd.concat(inner_rows, ignore_index=True)
outer_df = pd.DataFrame(outer_rows)
topfeat_df = pd.DataFrame(topfeat_rows)

inner_df.to_csv("inner_metrics.csv", index=False)
outer_df.to_csv("outer_metrics.csv", index=False)

outer_df[
    [
        "outer_fold",
        "auc",
        "tn",
        "fp",
        "fn",
        "tp",
        "sensitivity",
        "specificity",
        "balanced_accuracy",
    ]
].to_csv("outer_confusion_and_rates.csv", index=False)

topfeat_df.to_csv("top50_features_per_outer_fold.csv", index=False)

print("Mean metrics (outer folds):")
print(outer_df[["balanced_accuracy", "accuracy", "f1", "auc"]].mean())

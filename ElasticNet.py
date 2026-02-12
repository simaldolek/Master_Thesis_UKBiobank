import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, make_scorer

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


CSV_PATH = "/opt/notebooks/FinalFeatures/ICA_Pearson_Full_Features.csv"
TARGET_COL = "ptsd"
DROP_ID_COL = True          
DROP_LAST_DEMOG_COLS = 8    # last 8 columns are demographics - drop
PERCENTILE = 50             # keep top 50%

C_grid = [0.01, 0.1, 1, 10]
l1_grid = [0.1, 0.3, 0.5, 0.7, 0.9]

# load UKB data
df = pd.read_csv(CSV_PATH)

if DROP_ID_COL:
    df = df.iloc[:, 1:]  # drop ID column

y = df[TARGET_COL].astype(int).to_numpy()
X_df = df.drop(columns=[TARGET_COL])
if DROP_LAST_DEMOG_COLS > 0:
    X_df = X_df.iloc[:, :-DROP_LAST_DEMOG_COLS]

X = X_df.to_numpy(dtype=np.float32) # to save memory


def metrics_from_pred(y_true, y_pred):
    cm = confusion_matrix(y_true.astype(int), y_pred.astype(int), labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    acc = (y_pred == y_true).mean()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)  # sensitivity
    specificity = tn / (tn + fp + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    bal_acc = 0.5 * (recall + specificity)

    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "sensitivity": float(recall),
        "specificity": float(specificity),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

balacc_scorer = make_scorer(
    lambda yt, yp: metrics_from_pred(yt, yp)["balanced_accuracy"],
    greater_is_better=True
)


base_pipe = Pipeline(steps=[
    ("select", SelectPercentile(score_func=f_classif, percentile=PERCENTILE)),
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        max_iter=5000,
        tol=1e-3,              # for the large dataset
        class_weight=None,     
        random_state=RANDOM_STATE,
        n_jobs=-1,            
    ))
])

param_grid = {
    "clf__C": C_grid,
    "clf__l1_ratio": l1_grid
}

outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

inner_rows = []
outer_rows = []

for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Inner 
    search = GridSearchCV(
        estimator=base_pipe,
        param_grid=param_grid,
        scoring=balacc_scorer,
        cv=inner_cv,
        n_jobs=-1,
        refit=True,
        verbose=0,
        return_train_score=False
    )

    search.fit(X_train, y_train)


    cvres = pd.DataFrame(search.cv_results_)
    for _, row in cvres.iterrows():
        inner_rows.append({
            "outer_fold": outer_fold,
            "C": float(row["param_clf__C"]),
            "l1_ratio": float(row["param_clf__l1_ratio"]),
            "mean_balanced_accuracy": float(row["mean_test_score"]),
            "std_balanced_accuracy": float(row["std_test_score"]),
            "rank": int(row["rank_test_score"]),
        })

    best_C = float(search.best_params_["clf__C"])
    best_l1 = float(search.best_params_["clf__l1_ratio"])

    # Outer test
    best_model = search.best_estimator_
    y_pred_test = best_model.predict(X_test)
    mets = metrics_from_pred(y_test, y_pred_test)

    outer_rows.append({
        "outer_fold": outer_fold,
        "C": best_C,
        "l1_ratio": best_l1,
        **mets
    })


pd.DataFrame(inner_rows).to_csv("inner_metrics.csv", index=False)
pd.DataFrame(outer_rows).to_csv("outer_metrics.csv", index=False)
pd.DataFrame(outer_rows)[
    ["outer_fold", "tn", "fp", "fn", "tp", "sensitivity", "specificity", "balanced_accuracy"]
].to_csv("outer_confusion_and_rates.csv", index=False)

print("DONE")
print(pd.DataFrame(outer_rows)[["balanced_accuracy", "accuracy", "f1"]].mean())


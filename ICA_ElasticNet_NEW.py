import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix

torch.manual_seed(42)
np.random.seed(42)

CSV_PATH = "/opt/notebooks/FinalFeatures/ICA_Pearson_Full_Features.csv"

# LOAD DATA
df = pd.read_csv(CSV_PATH)

df = df.iloc[:, 1:]  # drop ID column

y = df["ptsd"].values.astype(np.float32)

X_df = df.drop(columns=["ptsd"])

# drop last 8 columns (demographics)
X_df = X_df.iloc[:, :-8]

X = X_df.values.astype(np.float32)

n_samples, n_features = X.shape


# HELPERS
def stratified_kfold_indices(y, k=5):
    y = np.array(y)
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]

    np.random.shuffle(idx0)
    np.random.shuffle(idx1)

    folds0 = np.array_split(idx0, k)
    folds1 = np.array_split(idx1, k)

    folds = []
    for i in range(k):
        folds.append(np.concatenate([folds0[i], folds1[i]]))
    return folds


def standardize_fit(X):
    mean = X.mean(0)
    std = X.std(0) + 1e-8
    return mean, std


def standardize_apply(X, mean, std):
    return (X - mean) / std


def corr_feature_select(X, y, frac=0.5):
    y_center = y - y.mean()
    X_center = X - X.mean(0)

    corr = (X_center * y_center[:, None]).mean(0) / (
        X_center.std(0) * y_center.std() + 1e-8
    )

    k = int(frac * X.shape[1])
    idx = np.argsort(-np.abs(corr))[:k]
    return idx



# MODEL
class ElasticNetLogReg(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        return self.linear(x).squeeze()


def train_model(X, y, C, l1_ratio, epochs=200, lr=0.01):
    X_t = torch.tensor(X)
    y_t = torch.tensor(y)

    model = ElasticNetLogReg(X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    l1_lambda = l1_ratio / C
    l2_lambda = (1 - l1_ratio) / C

    for _ in range(epochs):
        opt.zero_grad()

        preds = model(X_t)

        bce = nn.BCEWithLogitsLoss()(preds, y_t)

        l1 = torch.sum(torch.abs(model.linear.weight))
        l2 = torch.sum(model.linear.weight ** 2)

        loss = bce + l1_lambda * l1 + l2_lambda * l2

        loss.backward()
        opt.step()

    return model



# METRICS
def metrics(y_true, logits):
    # logits -> probabilities
    prob = 1.0 / (1.0 + np.exp(-logits))
    pred = (prob >= 0.5).astype(int)

    cm = confusion_matrix(y_true.astype(int), pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    acc = (pred == y_true).mean()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)  # sensitivity
    specificity = tn / (tn + fp + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    bal_acc = 0.5 * (recall + specificity)

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "sensitivity": recall,
        "specificity": specificity,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }



# NESTED CV
outer_folds = stratified_kfold_indices(y, 5)

C_grid = [0.01, 0.1, 1, 10]
l1_grid = [0.1, 0.3, 0.5, 0.7, 0.9]

inner_rows = []
outer_rows = []

for outer_i in range(5):

    test_idx = outer_folds[outer_i]
    train_idx = np.concatenate([outer_folds[j] for j in range(5) if j != outer_i])

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    inner_folds = stratified_kfold_indices(y_train, 5)

    best_score = -1
    best_params = None

    # INNER
    for C in C_grid:
        for l1r in l1_grid:

            scores = []

            for inner_i in range(5):
                val_idx = inner_folds[inner_i]
                tr_idx = np.concatenate([inner_folds[j] for j in range(5) if j != inner_i])

                X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
                X_val, y_val = X_train[val_idx], y_train[val_idx]

                # feature selection
                feat_idx = corr_feature_select(X_tr, y_tr, 0.5)

                X_tr = X_tr[:, feat_idx]
                X_val = X_val[:, feat_idx]

                # scaling
                m, s = standardize_fit(X_tr)
                X_tr = standardize_apply(X_tr, m, s)
                X_val = standardize_apply(X_val, m, s)

                model = train_model(X_tr, y_tr, C, l1r)

                logits = model(torch.tensor(X_val)).detach().numpy()

                mets = metrics(y_val, logits)
                scores.append(mets["balanced_accuracy"])

                inner_rows.append({
                    "outer_fold": outer_i,
                    "C": C,
                    "l1_ratio": l1r,
                    **mets
                })

            mean_score = np.mean(scores)

            if mean_score > best_score:
                best_score = mean_score
                best_params = (C, l1r)

    # OUTER TEST
    C_best, l1_best = best_params

    feat_idx = corr_feature_select(X_train, y_train, 0.5)

    X_tr = X_train[:, feat_idx]
    X_te = X_test[:, feat_idx]

    m, s = standardize_fit(X_tr)
    X_tr = standardize_apply(X_tr, m, s)
    X_te = standardize_apply(X_te, m, s)

    model = train_model(X_tr, y_train, C_best, l1_best)

    logits_test = model(torch.tensor(X_te)).detach().numpy()

    mets = metrics(y_test, logits_test)

    outer_rows.append({
        "outer_fold": outer_i,
        "C": C_best,
        "l1_ratio": l1_best,
        **mets
    })


pd.DataFrame(inner_rows).to_csv("inner_metrics.csv", index=False)
pd.DataFrame(outer_rows).to_csv("outer_metrics.csv", index=False)

# Save per-fold confusion matrix entries + sensitivity/specificity
pd.DataFrame(outer_rows)[["outer_fold", "tn", "fp", "fn", "tp", "sensitivity", "specificity"]].to_csv(
    "outer_confusion_and_rates.csv", index=False
)

print("DONE")

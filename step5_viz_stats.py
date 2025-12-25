import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
confusion_matrix, classification_report,
roc_curve, roc_auc_score,
precision_recall_curve, average_precision_score
)

# ---------- SETTINGS ----------
DATA_NPZ = "dataset_chb01_03_windows_LABELED.npz"
OUT_DIR = "reports"
RANDOM_STATE = 42
# -----------------------------

os.makedirs(OUT_DIR, exist_ok=True)

# Load dataset
d = np.load(DATA_NPZ, allow_pickle=True)
X = d["X"] # (n_windows, n_features)
y = d["y"].astype(int) # (n_windows,)
times = d.get("times", None) # optional
ch_names = d.get("ch_names", None)
bands = d.get("bands", None)

print("Loaded:", DATA_NPZ)
print("X:", X.shape, "y:", y.shape, "Positive labels:", int(y.sum()))

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# Train a simple baseline model
clf = LogisticRegression(max_iter=2000, class_weight="balanced")
clf.fit(X_train, y_train)

# Predictions
proba = clf.predict_proba(X_test)[:, 1]
pred = (proba >= 0.5).astype(int)

# ---- Metrics / Reports ----
cm = confusion_matrix(y_test, pred)
auc = roc_auc_score(y_test, proba)
ap = average_precision_score(y_test, proba)

print("\nConfusion matrix:\n", cm)
print("\nROC AUC:", auc)
print("Average Precision (PR AUC):", ap)
print("\nClassification report:\n", classification_report(y_test, pred, digits=3))

# ---- Plot: Confusion Matrix ----
plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks([0, 1], ["non-seiz", "seiz"])
plt.yticks([0, 1], ["non-seiz", "seiz"])
for (i, j), v in np.ndenumerate(cm):
    plt.text(j, i, str(v), ha="center", va="center")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"), dpi=200)
plt.close()

# ---- Plot: ROC Curve ----
fpr, tpr, _ = roc_curve(y_test, proba)
plt.figure()
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], linestyle="--")
plt.title(f"ROC Curve (AUC={auc:.3f})")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "roc_curve.png"), dpi=200)
plt.close()

# ---- Plot: Precision-Recall Curve ----
prec, rec, _ = precision_recall_curve(y_test, proba)
plt.figure()
plt.plot(rec, prec)
plt.title(f"Precision-Recall (AP={ap:.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "pr_curve.png"), dpi=200)
plt.close()

# ---- Statistics: feature-wise t-tests (seizure vs non-seizure) ----
# (This is a simple start; we can do better later with proper CV and effect sizes.)
X0 = X[y == 0]
X1 = X[y == 1]

# Guard in case labels are missing
if len(X1) < 2:
    print("\n⚠ Not enough seizure windows for stats. Check Step 4 seizure times.")
    raise SystemExit

t_stats, pvals = ttest_ind(X1, X0, axis=0, equal_var=False, nan_policy="omit")

# Multiple comparison correction (FDR)
rej, pvals_fdr, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")

stats_df = pd.DataFrame({
"feature_idx": np.arange(X.shape[1]),
"t_stat": t_stats,
"pval": pvals,
"pval_fdr": pvals_fdr,
"significant_fdr05": rej
}).sort_values("pval_fdr")

stats_path = os.path.join(OUT_DIR, "feature_stats_ttest_fdr.csv")
stats_df.to_csv(stats_path, index=False)

print(f"\nSaved plots to: {OUT_DIR}/")
print(f"Saved stats table to: {stats_path}")

# ---- Plot: top 20 most significant features ----
top = stats_df.head(20).copy()
plt.figure(figsize=(10, 5))
plt.bar(range(len(top)), -np.log10(top["pval_fdr"].clip(lower=1e-300)))
plt.title("Top 20 features by -log10(FDR p-value)")
plt.xlabel("Rank")
plt.ylabel("-log10(FDR p)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "top_features_fdr.png"), dpi=200)
plt.close()

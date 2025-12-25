import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix

DATA_NPZ = "../dataset_chb01_03_windows.npz"

# Load features
d = np.load(DATA_NPZ, allow_pickle=True)
X = d["X"] # (n_windows, n_features)

# IMPORTANT: labels are still dummy for now (we add real seizure labels later)
y = d["y"].astype(int)

print("Loaded:", DATA_NPZ)
print("X:", X.shape, "y:", y.shape, "Positive labels:", int(y.sum()))

# If y is all zeros, we can't train a classifier yet.
if len(np.unique(y)) < 2:
    print("\n⚠️ y has only one class (all 0).")
    print("That is expected right now because we haven't added seizure labels yet.")
    print("Next step is to create labels from seizure times.")
    raise SystemExit

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42, stratify=y
)

# Simple baseline model
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("\nAccuracy:", round(acc, 3))
print("AUC:", round(auc, 3))
print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred))
print("\nReport:\n", classification_report(y_test, y_pred, digits=3))
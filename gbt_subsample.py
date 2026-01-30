#!/usr/bin/env python
"""
Per-class subsample training: use a small number of samples per class for training,
test on the remaining samples. Supports Pressure / Photoelectric / Both views.
Useful to verify whether classes are linearly separable with very limited train data.
"""
import argparse
import pathlib
import re
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


def load_dataset(root: pathlib.Path):
    """Load pressure/photoelectric/both features and labels from txt files."""
    pattern = re.compile(r"(\d+)%25moisture\+(\d+)%25light")
    files = sorted((root / "soil calculation").glob("*.txt"))
    combos = sorted({(int(pattern.search(f.name).group(1)), int(pattern.search(f.name).group(2))) for f in files})
    label_map = {c: i for i, c in enumerate(combos)}  # 36 classes

    press_list, photo_list, both_list, labels = [], [], [], []
    for f in files:
        m = pattern.search(f.name)
        if not m:
            continue
        vals = [float(x) for x in f.read_text().split() if x.strip()]
        if len(vals) % 2 != 0:
            raise ValueError(f"odd length in {f}")
        data = np.array(vals, dtype=np.float32).reshape(-1, 2)  # [N,2]
        press_list.append(data[:, 0][:, None])   # [N,1]
        photo_list.append(data[:, 1][:, None])   # [N,1]
        both_list.append(data)                   # [N,2]
        labels.append(np.full(len(data), label_map[(int(m.group(1)), int(m.group(2)))], dtype=np.int64))

    X_press = np.vstack(press_list)
    X_photo = np.vstack(photo_list)
    X_both = np.vstack(both_list)
    y = np.concatenate(labels)
    return X_press, X_photo, X_both, y


def subsample_split(X: np.ndarray, y: np.ndarray, per_class: int, seed: int):
    """Per-class subsample for train; remaining samples go to test."""
    rng = np.random.default_rng(seed)
    train_idx = []
    test_idx = []
    classes = np.unique(y)
    for c in classes:
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        train_idx.extend(idx[:per_class])
        test_idx.extend(idx[per_class:])
    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def run_view(name: str, X: np.ndarray, y: np.ndarray, per_class: int, seed: int):
    X_train, X_test, y_train, y_test = subsample_split(X, y, per_class, seed)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=6,
        max_iter=300,
        random_state=seed
    )
    clf.fit(X_train_s, y_train)
    y_pred = clf.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=range(len(np.unique(y))))

    print(f"\n== {name} (train {per_class}/class, total train {len(y_train)}) ==")
    print(f"Accuracy: {acc:.4f}")
    print("Confusion matrix shape:", cm.shape)
    print(cm)


def main():
    parser = argparse.ArgumentParser(description="Per-class subsample GBDT baseline")
    parser.add_argument("--data-root", default=".", help="project root containing 'soil calculation'")
    parser.add_argument("--per-class-train", type=int, default=50, help="train samples per class")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    args = parser.parse_args()

    root = pathlib.Path(args.data_root).expanduser().resolve()
    X_press, X_photo, X_both, y = load_dataset(root)
    print(f"Loaded data: {X_both.shape[0]} samples, {X_both.shape[1]} features, classes: {len(np.unique(y))}")

    run_view("Pressure_only", X_press, y, args.per_class_train, args.seed)
    run_view("Photoelectric_only", X_photo, y, args.per_class_train, args.seed)
    run_view("Both_channels", X_both, y, args.per_class_train, args.seed)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Gradient boosting (tree) baseline for the soil dataset.
- Reads all txt files under "soil calculation" (server filenames use %25 for '%').
- Each txt has two interleaved channels; we treat each (pressure, photoelectric) pair as one sample.
- Builds three views: Pressure-only (1 feature), Photoelectric-only (1 feature), Both (2 features).
- StandardScaler is fitted per-view to handle scale differences, then a HistGradientBoostingClassifier is trained/evaluated.
- Prints accuracy and confusion matrix for each view.
"""
import argparse
import pathlib
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score


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


def run_view(name: str, X: np.ndarray, y: np.ndarray, test_size: float, seed: int):
    # Stratified split by label (samples treated as independent)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=6,
        max_iter=300,
        l2_regularization=0.0,
        random_state=seed
    )
    clf.fit(X_train_s, y_train)
    y_pred = clf.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=range(len(np.unique(y))))

    print(f"\n== {name} ==")
    print(f"Accuracy: {acc:.4f}")
    print("Confusion matrix shape:", cm.shape)
    print(cm)


def main():
    parser = argparse.ArgumentParser(description="Gradient boosting baseline on soil dataset")
    parser.add_argument("--data-root", default=".", help="project root containing 'soil calculation'")
    parser.add_argument("--test-size", type=float, default=0.2, help="test split ratio")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    args = parser.parse_args()

    root = pathlib.Path(args.data_root).expanduser().resolve()
    X_press, X_photo, X_both, y = load_dataset(root)
    print(f"Loaded data: {X_both.shape[0]} samples, {X_both.shape[1]} features, labels: {len(np.unique(y))}")

    run_view("Pressure_only", X_press, y, args.test_size, args.seed)
    run_view("Photoelectric_only", X_photo, y, args.test_size, args.seed)
    run_view("Both_channels", X_both, y, args.test_size, args.seed)


if __name__ == "__main__":
    main()

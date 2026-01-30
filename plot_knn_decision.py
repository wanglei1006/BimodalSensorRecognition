#!/usr/bin/env python
"""
KNN decision regions + train/test scatter for soil calculation (2D features).
- Feature1: Pressure, Feature2: Photoelectric
- Colors: classes; markers: train/test
Outputs: knn_decision.png
"""
import pathlib
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


def load_dataset(root: pathlib.Path):
    pat = re.compile(r"(\d+)%25?moisture\+(\d+)%25?light")
    files = sorted((root / "soil calculation").glob("*.txt"))
    combos = []
    matched = []
    for f in files:
        m = pat.search(f.name)
        if m:
            combos.append((int(m.group(1)), int(m.group(2))))
            matched.append(f)
    combos = sorted(set(combos))
    if not combos:
        raise RuntimeError("No files matched pattern '*%moisture+*%light.txt'.")
    label_map = {c: i for i, c in enumerate(combos)}

    X_list = []
    y_list = []
    for f in matched:
        m = pat.search(f.name)
        vals = [float(x) for x in f.read_text().split() if x.strip()]
        if len(vals) % 2 != 0:
            raise ValueError(f"odd length in {f}")
        data = np.array(vals, dtype=np.float32).reshape(-1, 2)  # [N,2]
        X_list.append(data)
        y_list.append(np.full(len(data), label_map[(int(m.group(1)), int(m.group(2)))]))
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    return X, y


def main():
    root = pathlib.Path(".").resolve()
    # Use Arial Narrow if available on server
    _arial_path = "/usr/share/fonts/truetype/arial/ARIALN.TTF"
    _font_prop = None
    if pathlib.Path(_arial_path).exists():
        font_manager.fontManager.addfont(_arial_path)
        _font_prop = font_manager.FontProperties(fname=_arial_path, weight="bold")
    X, y = load_dataset(root)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    # model for plotting decision regions (larger k + uniform weight for smoother regions)
    knn_plot = KNeighborsClassifier(n_neighbors=25, weights="uniform", metric="manhattan")
    knn_plot.fit(X_tr_s, y_tr)

    # decision grid
    x_min, x_max = X_tr_s[:, 0].min() - 0.5, X_tr_s[:, 0].max() + 0.5
    y_min, y_max = X_tr_s[:, 1].min() - 0.5, X_tr_s[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 150),
                         np.linspace(y_min, y_max, 150))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = knn_plot.predict(grid).reshape(xx.shape)

    # pick a few samples per class for visualization
    rng = np.random.default_rng(42)
    def sample_idx(y, per_class):
        idxs = []
        for c in np.unique(y):
            candidates = np.where(y == c)[0]
            take = min(per_class, len(candidates))
            idxs.extend(rng.choice(candidates, size=take, replace=False))
        return np.array(idxs, dtype=int)

    tr_idx = sample_idx(y_tr, per_class=1)
    te_idx = sample_idx(y_te, per_class=1)
    # reduce test samples by half for cleaner view
    if len(te_idx) > 1:
        te_idx = rng.choice(te_idx, size=max(1, len(te_idx)//2), replace=False)

    plt.figure(figsize=(12, 4))
    plt.pcolormesh(xx, yy, Z, alpha=0.25, cmap="tab20", shading="nearest")
    # small horizontal jitter to separate train/test markers visually
    plt.scatter(X_tr_s[tr_idx, 0] - 0.03, X_tr_s[tr_idx, 1], c=y_tr[tr_idx], cmap="tab20",
                s=160, marker="o", alpha=0.9, label="train")
    plt.scatter(X_te_s[te_idx, 0], X_te_s[te_idx, 1],
                c="#4F6D7A", s=190, marker="^", alpha=0.9, label="test")
    plt.legend(loc="upper right", fontsize=19, prop=_font_prop)
    if _font_prop:
        plt.xlabel("Feature 1 (Pressure, z-score)", fontproperties=_font_prop, fontsize=20, fontweight="bold")
        plt.ylabel("Feature 2 (Photoelectric, z-score)", fontproperties=_font_prop, fontsize=20, fontweight="bold")
        plt.title("kNN decision regions with train/test scatter", fontproperties=_font_prop, fontsize=20, fontweight="bold")
    else:
        plt.xlabel("Feature 1 (Pressure, z-score)", fontsize=20, fontweight="bold")
        plt.ylabel("Feature 2 (Photoelectric, z-score)", fontsize=20, fontweight="bold")
        plt.title("kNN decision regions with train/test scatter", fontsize=20, fontweight="bold")
    plt.tight_layout()
    out = root / "knn_decision.png"
    plt.savefig(out, dpi=200)
    print("saved", out)


if __name__ == "__main__":
    main()

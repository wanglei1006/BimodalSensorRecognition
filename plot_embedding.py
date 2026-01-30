#!/usr/bin/env python
"""
Project 2D embedding of normalized samples (both channels) and color by class.
- Reads txt under "soil calculation" (supports % or %25 in filenames).
- Each odd/even row pair is one sample: [pressure, photoelectric].
- StandardScaler on full dataset, then PCA to 2D for visualization.
- Saves scatter plot to embed_scatter.png.
"""
import pathlib
import re
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# Use Arial Narrow if available on server
_arial_path = "/usr/share/fonts/truetype/arial/ARIALN.TTF"
_font_prop = None
if pathlib.Path(_arial_path).exists():
    font_manager.fontManager.addfont(_arial_path)
    _font_prop = font_manager.FontProperties(fname=_arial_path, weight="bold")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial Narrow"]


def load_dataset(root: pathlib.Path):
    pat = re.compile(r"(\d+)%25?moisture\+(\d+)%25?light")
    files = sorted((root / "soil calculation").glob("*.txt"))
    combos = []
    matched_files = []
    for f in files:
        m = pat.search(f.name)
        if m:
            combos.append((int(m.group(1)), int(m.group(2))))
            matched_files.append(f)
    combos = sorted(set(combos))
    if not combos:
        raise RuntimeError("No files matched pattern '*%moisture+*%light.txt' (with or without %25).")
    label_map = {c: i for i, c in enumerate(combos)}  # 36 classes

    X_list = []
    y_list = []
    for f in matched_files:
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
    X, y = load_dataset(root)

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_s)

    plt.figure(figsize=(12, 4))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, s=1, cmap="tab20", alpha=0.7)
    # label each class once at the class centroid
    for cls in np.unique(y):
        mask = y == cls
        cx = X_2d[mask, 0].mean()
        cy = X_2d[mask, 1].mean()
        # place label with a small offset from centroid
        plt.text(
            cx + 0.03,
            cy + 0.03,
            str(int(cls) + 1),
            fontsize=22,
            alpha=0.9,
            fontweight="bold",
            fontproperties=_font_prop,
        )
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    out = root / "embed_scatter.png"
    plt.savefig(out, dpi=200)
    print("saved", out)


if __name__ == "__main__":
    main()

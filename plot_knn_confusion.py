#!/usr/bin/env python
"""
Train a kNN on both channels and plot a confusion matrix with per-class precision/recall/F1.
- Assumes data under "soil calculation" with filenames like 20%25moisture+0%25light.txt.
- Each file has interleaved channels: odd rows = pressure, even rows = photoelectric.
"""
import pathlib
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import seaborn as sns


def load_dataset(root: pathlib.Path):
    # 支持文件名 20%moisture+0%light.txt 或 20%25moisture+0%25light.txt
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
        raise RuntimeError("No files matched pattern '*%moisture+*%light.txt' (with or without %25)")
    label_map = {c: i for i, c in enumerate(combos)}  # 36 classes

    both_list = []
    labels = []
    for f in matched_files:
        m = pat.search(f.name)
        vals = [float(x) for x in f.read_text().split() if x.strip()]
        if len(vals) % 2 != 0:
            raise ValueError(f"odd length in {f}")
        data = np.array(vals, dtype=np.float32).reshape(-1, 2)  # [N,2]
        both_list.append(data)
        labels.append(np.full(len(data), label_map[(int(m.group(1)), int(m.group(2)))]))

    X = np.vstack(both_list)
    y = np.concatenate(labels)
    return X, y, combos


def main():
    root = pathlib.Path(".").resolve()
    # Use Arial Narrow if available on server
    _arial_path = "/usr/share/fonts/truetype/arial/ARIALN.TTF"
    _font_prop = None
    if pathlib.Path(_arial_path).exists():
        font_manager.fontManager.addfont(_arial_path)
        _font_prop = font_manager.FontProperties(fname=_arial_path)
    try:
        X, y, combos = load_dataset(root)
    except RuntimeError:
        # fallback to parent directory (when running inside soil-calculation-code)
        X, y, combos = load_dataset(root.parent)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = KNeighborsClassifier(n_neighbors=5, weights="distance")
    clf.fit(X_train_s, y_train)
    y_pred = clf.predict(X_test_s)

    cm = confusion_matrix(y_test, y_pred, labels=range(len(combos)))
    # normalize CM rows for display
    disp = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    # annotate all non-zero values, show as percent with 1 decimal
    annot = np.empty(disp.shape, dtype=object)
    annot[:] = ""
    for i in range(disp.shape[0]):
        for j in range(disp.shape[1]):
            val = disp[i, j]
            if np.isnan(val):
                continue
            if val > 0:
                pct = val * 100
                if abs(pct - round(pct)) < 1e-6:
                    annot[i, j] = f"{int(round(pct))}"
                else:
                    annot[i, j] = f"{pct:.1f}"

    fig, ax = plt.subplots(figsize=(12, 6))
    n = disp.shape[0]
    hm = sns.heatmap(
        disp,
        cmap="Blues",
        ax=ax,
        vmin=0,
        vmax=1,
        annot=annot,
        fmt="",
        cbar=True,
        mask=np.isnan(disp),
        xticklabels=[str(i) for i in range(1, n + 1)],
        yticklabels=[str(i) for i in range(1, n + 1)],
        annot_kws={"size": 11, "fontproperties": _font_prop} if _font_prop else {"size": 11},
        linewidths=0.3,
        linecolor="#E6E6E6",
    )
    # colorbar in percent (0-100)
    cbar = hm.collections[0].colorbar
    cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cbar.set_ticklabels(["0", "20", "40", "60", "80", "100"])
    cbar.ax.tick_params(labelsize=14)
    if _font_prop:
        ax.set_title("kNN Confusion Matrix (Both Channels)\n(values in %)", fontproperties=_font_prop, fontsize=16)
        ax.set_xlabel("Predicted label", fontproperties=_font_prop, fontsize=16)
        ax.set_ylabel("True label", fontproperties=_font_prop, fontsize=16)
        for lab in ax.get_xticklabels() + ax.get_yticklabels():
            lab.set_fontproperties(_font_prop)
        for lab in cbar.ax.get_yticklabels():
            lab.set_fontproperties(_font_prop)
    else:
        ax.set_title("kNN Confusion Matrix (Both Channels)\n(values in %)", fontsize=16)
        ax.set_xlabel("Predicted label", fontsize=16)
        ax.set_ylabel("True label", fontsize=16)
    ax.tick_params(axis="x", rotation=90, labelsize=14)
    ax.tick_params(axis="y", rotation=0, labelsize=14)
    plt.tight_layout()
    out = root / "knn_confusion.png"
    fig.savefig(out, dpi=200)
    print("saved", out)


if __name__ == "__main__":
    main()

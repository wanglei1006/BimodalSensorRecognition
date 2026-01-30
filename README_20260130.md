# Tabular Feature Baseline (Gradient Boosting Trees)

## Virtual Environment Setup
From the project root (the directory containing `requirements.txt`):
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Dependencies (from requirements.txt)
Key packages (see `requirements.txt` for the full list and exact versions):
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
- scipy
- torch
- torchvision
- opencv-python

## Model and Preprocessing
- Features: each sample is 2D (pressure, photoelectric), with 36 classes (humidity × light combinations).
- Data loading: each txt alternates two signal channels; pair them as (pressure, photoelectric) for one sample; 36 files in total with about 36k samples.
- Preprocessing: `StandardScaler` is fitted independently per view to handle scale differences.
- Model: `HistGradientBoostingClassifier` (scikit-learn).
  - Main params: `learning_rate=0.05, max_depth=6, max_iter=300, l2_regularization=0.0, random_state=seed`
  - Views: Pressure_only, Photoelectric_only, Both_channels.

## Run
Run from the server project root (contains the “soil calculation” directory). Directory format:
```
~/materials/
├── soil calculation/
└── soil calculation code/
```
```bash
cd ~/materials
python tabular_gbt.py \
  --data-root . \
  --test-size 0.2 \
  --seed 42
```

## Subsampling Experiment (few samples per class for training)
- Purpose: use only a small number of samples per class (default 50) for training and the rest for testing, to check class separability.
- Run:
```bash
cd ~/materials
python gbt_subsample.py \
  --data-root . \
  --per-class-train 50 \
  --seed 42
```

## Simple Baselines (Logistic Regression / kNN)
Used for sanity checks: if a simple model already scores high, the data is naturally separable; if it drops significantly, the complex model is capturing the distribution.
```bash
cd ~/materials
python simple_baselines.py \
  --data-root . \
  --test-size 0.2 \
  --seed 42
```

## KNN Method Notes
This method is a non-parametric classifier based on distance metrics. The two-channel input features are standardized so signals with different scales are comparable, ensuring stable and interpretable distance calculations. Then, in the sample space, the K nearest neighbors to a query sample are selected, and a distance-weighted strategy strengthens the influence of closer neighbors to improve boundary-sample discrimination. With this design, the KNN process is intuitive and well-structured, enabling signal class recognition without complex training, and it has shown stable results in practice.

## kNN Confusion Matrix Image (two channels, with precision/recall/F1 sidebars)
```bash
cd ~/materials
MPLCONFIGDIR=/tmp/mpl python plot_knn_confusion.py
```
Generates `knn_confusion.png`.

## kNN Decision Region Plot (train/test scatter + decision regions)
Based on the 2D features (Pressure/Photoelectric) in `soil calculation`, plot kNN decision regions and label a small number of train/test samples:
```bash
cd ~/materials
MPLCONFIGDIR=/tmp/mpl python plot_knn_decision.py
```
Generates `knn_decision.png`.

## PCA Distribution Plot (plot_embedding.py)
After z-score standardization of the two-channel samples, reduce to 2D with PCA and output a scatter plot colored by class:
```bash
cd ~/materials
MPLCONFIGDIR=/tmp/mpl python plot_embedding.py
```
Generates `embed_scatter.png`.

## Output
- Console prints Accuracy and confusion matrix (36×36) for each view.
- To change splits, adjust `--test-size` and `--seed`. To save confusion matrices, add `np.save` in the scripts.

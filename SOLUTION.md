# Solution Report

## Reproducibility

Run from the repository root:

```bash
python3 -m pip install -r requirements.txt
python3 solution.py
```

The command loads `Qwen/Qwen2.5-0.5B`, extracts hidden states for
`data/dataset.csv` and `data/test.csv`, evaluates the probe on the labelled
data, writes `results.json`, and writes competition predictions to
`predictions.csv`.

## Final Approach

The solution keeps the fixed infrastructure unchanged and modifies only the
student files:

- `aggregation.py`: uses several late transformer layers instead of only the
  final layer. For each selected layer it concatenates the last-token vector,
  the mean over the last 64 real tokens, the mean over all real tokens, and the
  standard deviation over the last 64 real tokens.
- `probe.py`: replaces the small MLP training loop with a deterministic
  regularized, class-balanced logistic regression probe over the top 2048
  univariate hidden-state features. The validation split is still used to tune
  the final decision threshold.
- `splitting.py`: uses shuffled stratified k-fold splits with a stratified
  validation subset inside each fold, reducing dependence on a single random
  holdout split.

This setup is lightweight, deterministic, and better suited to the small
labelled dataset than an unconstrained neural probe.

## Current Results

The latest official `python3 solution.py` run produced:

- Folds: 7
- Feature dimension: 17920
- Average validation AUROC: 0.7600
- Average test accuracy: 0.7518
- Average test F1: 0.8239
- Average test AUROC: 0.7589

## Experiments And Notes

The starting baseline used only the last real token from the final transformer
layer and a small MLP. That setup is sensitive to the exact split and can
overfit the small dataset. The final implementation favors regularized linear
models over richer pooled hidden-state features, which gave a more stable
evaluation while keeping runtime low.

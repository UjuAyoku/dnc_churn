# Deep & Cross Network (DCN) Reimplementation for Customer Churn

A from-scratch TensorFlow/Keras reimplementation of the cross layer from
**Wang et al., "Deep & Cross Network for Ad Click Predictions" (ADKDD 2017,
[arXiv:1708.05123](https://arxiv.org/abs/1708.05123))**, evaluated on the Telco
customer churn dataset against logistic regression, random forest and XGBoost.

## What the paper proposes

Tabular models with sparse categorical features usually learn feature
interactions either by hand-crafting them or implicitly through a deep MLP. DCN
adds a **cross network** that builds explicit feature crosses of increasing
degree. Each cross layer computes

```
x_{l+1} = x_0 * (x_l^T w_l) + b_l + x_l
```

where `x_0` is the input vector (embeddings + numeric features), so `L` layers
capture interactions up to degree `L + 1` with only `O(L * d)` extra parameters.
The cross network runs in parallel with a standard deep network, and their
outputs are concatenated before the final prediction.

## What I implemented

- `CrossLayer`: a custom Keras layer implementing the equation above
  (`dcn_churn.py`).
- A DCN model: learned embeddings for the 15 categorical columns, concatenated
  with 4 standardized numeric features, then a 3-layer cross network in parallel
  with a 64-32 MLP.
- Two ablations: **cross-only** and **deep-only (MLP with embeddings)**.
- Baselines: logistic regression, random forest, XGBoost (one-hot features).
- A unit test checking the layer's output against a direct NumPy evaluation of
  the paper's equation (`test_cross_layer.py`).

## Experimental setup

| Item | Setting |
|---|---|
| Data | Telco customer churn, 7,043 customers, 26.5% churn, 15 categorical + 4 numeric features |
| Evaluation | Stratified 5-fold cross-validation, 1 repeat, seed 42 |
| Leakage control | Encoders and scalers fit on the training fold only; early stopping uses a 15% slice of the training fold, never the test fold |
| Metrics | ROC-AUC and PR-AUC (average precision; useful because of class imbalance) |
| Neural nets | Adam (lr 1e-3), batch size 256, early stopping (patience 10), dropout 0.2, L2 1e-5 |
| Software | TensorFlow 2.21.0, scikit-learn 1.8.0, XGBoost 3.4.1, pandas 3.0.2 |

## Results

Mean ± std across 5 folds (`results/summary.csv`, per-fold numbers in
`results/fold_results.csv`):

| Model | ROC-AUC | PR-AUC |
|---|---|---|
| Logistic regression | 0.8451 ± 0.0150 | 0.6567 ± 0.0296 |
| Random forest | 0.8411 ± 0.0145 | 0.6484 ± 0.0341 |
| XGBoost | 0.8452 ± 0.0118 | 0.6627 ± 0.0219 |
| Deep-only (MLP + embeddings) | 0.8440 ± 0.0140 | 0.6577 ± 0.0240 |
| Cross-only | 0.8417 ± 0.0171 | 0.6518 ± 0.0357 |
| **DCN (cross + deep)** | **0.8449 ± 0.0151** | **0.6614 ± 0.0284** |

Paired per-fold ROC-AUC differences, DCN minus each model (mean ± std):

| Comparison | Difference |
|---|---|
| vs. deep-only | +0.0009 ± 0.0017 |
| vs. cross-only | +0.0032 ± 0.0022 |
| vs. random forest | +0.0037 ± 0.0026 |
| vs. logistic regression | -0.0002 ± 0.0025 |
| vs. XGBoost | -0.0003 ± 0.0045 |

## Findings

1. **DCN performs on par with strong baselines but does not beat them.** All
   six models land within about 0.004 ROC-AUC of each other, well inside the
   ~0.015 fold-to-fold spread. With five folds I would not claim that any of
   these small gaps is real.
2. **The combination helps relative to its parts.** DCN beat cross-only in all
   5 folds and was ahead of deep-only in 4 of 5, but the margins are tiny
   (+0.003 and +0.001).
3. **A well-specified logistic regression is a very strong baseline here.** On
   a small dataset dominated by low-cardinality categorical features, there is
   little for explicit feature crossing to find.
4. **Cost:** DCN took ~14.5 s per fold versus ~0.0 s for logistic regression,
   for no measurable accuracy gain on this dataset.

A plausible reading, which I have not tested, is that the paper's gains come
from large-scale, high-cardinality sparse data (the paper evaluates on Criteo
ad-click data), where hand-crafting crosses is impractical. Testing that would
require a dataset of that kind.

## Differences from the paper

- Embedding sizes use the fast.ai heuristic `min(50, round(1.6 * card^0.56))`,
  because the paper's rule targets high-cardinality ad features and would
  over-size 2-4 category columns.
- No batch normalization or gradient clipping in the deep network. Check these
  against the paper's experimental setup before comparing numbers.
- Only the original cross layer (a vector `w` per layer) is implemented, not
  the later DCN-V2 matrix formulation.
- Fixed architecture (3 cross layers, 64-32 MLP) with no hyperparameter search.
  Results could shift with tuning.

## Limitations

- Single dataset, 5 folds, 1 repeat, one seed. The default `--repeats 3` gives
  15 evaluations and tighter estimates but takes longer on CPU.
- Folds share training data, so I report paired differences descriptively
  rather than with significance tests.
- Neural-net results can vary slightly between TensorFlow versions and
  hardware, even with seeds set.

## Reproduce

```bash
pip install -r requirements.txt
# the Telco CSV is in the churn-predictor repo; place it in this folder
python -m pytest -q                      # verifies the cross layer
python dcn_churn.py --data WA_Fn-UseC_-Telco-Customer-Churn.csv --folds 5 --repeats 1
```

Outputs are written to `results/` (`summary.csv`, `fold_results.csv`,
`run_meta.json`).

## Possible extensions

- Add DCN-V2 and compare against the original cross layer.
- Run the same code on a large sparse dataset (e.g., Criteo) to test whether the
  paper's gains appear at scale.
- Tune cross depth, embedding size and regularization with nested CV.
- Add SHAP or cross-weight inspection to see which feature interactions the
  cross network emphasizes.

## Reference

Wang, R., Fu, B., Fu, G., & Wang, M. (2017). *Deep & Cross Network for Ad Click
Predictions.* Proceedings of the ADKDD'17. https://arxiv.org/abs/1708.05123

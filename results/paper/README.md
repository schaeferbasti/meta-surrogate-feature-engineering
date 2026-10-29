# Paper figures (ESANN 2027): which part of meta-learned feature engineering fails

All numbers are in `numbers.json`. Regenerate everything from finished results (minutes, no cluster):

```bash
export SURROGATE_CHECK_KB=lightgbm_cv SURROGATE_CHECK_DIR=results/surrogate_check_cv PYTHONPATH=.
A=/work/dlclarge1/matusd-schaefeb-metafe/venvs/analysis/bin/python
$A experiments/surrogate_check/within_dataset.py --workers 14
$A experiments/surrogate_check/learning_curve.py --workers 14
$A experiments/surrogate_check/dataset_selection.py --workers 14   # ~7 min (500 permutations)
/work/dlclarge1/matusd-schaefeb-metafe/venvs/ag12/bin/python experiments/paper/figures.py   # needs matplotlib
```

Setup: 44 OpenML datasets (classification and regression), LightGBM as the target model, test error = log loss or
RMSE on the fixed OpenML split, mean over 3 seeds. Knowledge base (KB): 119,261 one-feature candidates, each labelled
with the relative improvement of the validation loss when the candidate is added to the original data.

## Figures and draft captions

**Fig. 1 `fig1_label_reliability`: the labels of the original KB are mostly noise.** Two independent measurements of
the label of the same candidates, as per-dataset ranks. (a) AutoGluon LightGBM (8-fold bagging), the label of the
original KB, with two seeds: Spearman 0.27, ICC 0.29 (3 datasets, 147 candidates; with the knowledge-base seed 0
included: ICC 0.23). About half of the labels are exactly 0, hence the ties. (b) Plain LightGBM with 5-fold CV,
mean of repeats 1-5 vs. 6-10 (each half is noisier than the 10-repeat label in the new KB): Spearman 0.67; reliability
of the 10-repeat label ICC 0.83 (median over 44 datasets, 35 of 44 >= 0.5).

**Fig. 2 `fig2_fe_comparison` (+ `table_fe_comparison.tex`): MetaFE does not improve on the original data.** Test error
decrease per dataset. TabPrep median +3.7 % (p = 0.006), OpenFE +1.1 % (36 datasets, p = 0.04), MetaFE 0.0 % at every
time limit; at 1000 s MetaFE is significantly worse than the original data (mean -1.3 %, p = 0.04). Symlog axis.

**Fig. 3 `fig3_metafe_picks`: MetaFE's surrogate is overconfident and its picks are random.** (a) Predicted vs. true
(CV label) improvement of the first feature MetaFE adds (37 datasets; 7 runs added none): predicted median +2.9 %,
true -0.09 %, positive on 11 of 37. (b) Percentile of that feature among all candidates of the dataset: median 45,
flat = random. Every later step is also predicted >= 1 % (median 4.9 %), so MetaFE does not stop and runs into the
time limit (median 25.5 steps).

**Fig. 4 `fig4_transfer_gap`: the core result.** Spearman of the predicted and the true label of the held-out
dataset's candidates. With the AutoGluon labels no surrogate ranks (0.02). With reliable labels, MetaFE's `describe()`
metafeatures still fail (0.08); target-aware metafeatures (mutual information, correlation ratio with the target and
with the residual of a LightGBM, gains over the operands) reach 0.31 when trained on the other 43 datasets, but 0.74
when trained on other candidates of the same dataset (grouped by first operand), close to the label-noise ceiling 0.89.
Neither labels nor metafeatures are the bottleneck; the transfer across datasets is. Top-10 picks above the
dataset's mean label: other datasets +0.24 %, same dataset +1.39 %, oracle +2.32 %.

**Fig. 5 `fig5_learning_curve`: 25 labels of the new dataset replace the KB of 43 datasets.** Surrogate trained on n
labelled candidates of the dataset itself (test: 30 % of its candidates; mean of 5 draws, median and IQR over
datasets; fewer datasets for n >= 100: 42/38/33). The KB surrogate as an additional input adds <= 0.015 at every n.

**Fig. 6 `fig6_dataset_selection`: dataset-level meta-learning does not help either.** Choosing the FE method per
dataset with a random forest on 9 dataset metafeatures (LODO), vs. fixed choices, the best validation score and the
oracle. (a) Original or TabPrep: meta +3.9 % mean (permutation p = 0.36) vs. always TabPrep +4.4 %; the validation
score +7.6 % (it avoids TabPrep's failures on kr_vs_kp and shuttle). (b) With OpenFE (36 datasets): meta +2.7 %
(p = 0.21); the validation score picks OpenFE on click_pred and okcupid, where it fails on test (-4.1 %).

## Caveats to state in the paper
- One target model (LightGBM), 44 datasets, 3 seeds.
- Fig. 1a rests on 3 datasets (the AutoGluon rerun is expensive); the KB-wide reliability of the new label (Fig. 1b)
  is on all 44.
- The same-dataset split groups by the first operand only; candidates can still share their second operand.
- Fig. 6 has 44 rows; the gap to the oracle is mostly 4 datasets where TabPrep fails (kr_vs_kp, shuttle, boston,
  satellite).

`metafe_first_picks.csv`: MetaFE's first pick per dataset (feature, predicted, true label, percentile, steps).

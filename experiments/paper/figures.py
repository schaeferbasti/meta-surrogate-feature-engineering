"""Figures and tables of the paper (ESANN 2027): which part of meta-learned feature engineering fails.

Reads only finished results (no model fit):
  results/label_reliability/{ag,candidates}/          AutoGluon label rerun with seeds 1-3 (label reliability test)
  data/lightgbm_cv/label_repeats/GBM/                 per-repeat CV labels of the new knowledge base
  data/lightgbm_cv/metafeature_matrix.parquet         new knowledge base (labels)
  data/lightgbm_cv/fe/MetaFE_GBM/*_log.csv            MetaFE steps with predicted improvement
  results/lightgbm_cv/evaluation/GBM/                 test/validation scores of all FE methods (3 seeds)
  results/surrogate_check/step1_lodo/                 LODO on the old (AutoGluon) labels
  results/surrogate_check_cv/{step1_lodo,within_dataset,learning_curve,dataset_selection}/   new labels

    python experiments/paper/figures.py      (any venv with matplotlib, e.g. venvs/ag12)
Output: results/paper/fig{1..6}_*.{pdf,png}, table_*.tex, numbers.json
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr, wilcoxon  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments" / "label_reliability"))
from metafe import config  # noqa: E402
from label_reliability import icc1, spearman_brown  # noqa: E402

OUT = REPO / "results" / "paper"
KB = REPO / "data" / "lightgbm_cv"
REL = REPO / "results" / "label_reliability"
CHECK_OLD = REPO / "results" / "surrogate_check"
CHECK = REPO / "results" / "surrogate_check_cv"
EVAL = REPO / "results" / "lightgbm_cv" / "evaluation" / "GBM"
TIME_LIMITS = [100, 250, 500, 1000]

# Palette (categorical order fixed; validated reference palette of the dataviz guide), ink and grid
BLUE, ORANGE, AQUA, YELLOW, VIOLET, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6", "#e34948"
INK, INK2, GRID, NEUTRAL = "#0b0b0b", "#52514e", "#e4e3df", "#8a8984"
FULL_W, HALF_W = 4.8, 2.35  # inches (ESANN text width ~12.2 cm)

plt.rcParams.update({
    "font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 6.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.axisbelow": True, "lines.linewidth": 1.5, "pdf.fonttype": 42, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "legend.frameon": False})


def save(fig, name):
    for ext in ["pdf", "png"]:
        fig.savefig(OUT / f"{name}.{ext}", dpi=300)
    plt.close(fig)


def pct_rank(s, by):
    return s.groupby(by).rank(pct=True)


# ------------------------------------------------------------------------------------------------------------- data

def kb_labels():
    kb = pd.read_parquet(KB / "metafeature_matrix.parquet", columns=["dataset - id", "feature - name", "operator",
                                                                     "improvement"])
    return kb[kb["operator"] != "delete"]


def fe_gains():
    """Gain (%) over Original per dataset (rows) and method, test and validation (mean over the 3 seeds)."""
    df = pd.concat([pd.read_parquet(f) for f in sorted(EVAL.glob("*_Result_*_*.parquet"))])
    mean = df.groupby(["dataset", "origin"])[["score_val", "score_test"]].mean().unstack("origin")
    out = {}
    for split in ["val", "test"]:
        err = -mean[f"score_{split}"]
        out[split] = err.drop(columns="Original").apply(lambda c: 100 * (err["Original"] - c) / err["Original"])
    return out


def metafe_first_picks(kb):
    rows = []
    for d in config.DATASETS:
        log = pd.read_csv(KB / "fe" / "MetaFE_GBM" / f"FE_{d}_MetaFE_log.csv")
        steps = log[log["step"].astype(str).str.fullmatch(r"[1-9]\d*")]
        k = kb[kb["dataset - id"] == d].set_index("feature - name")["improvement"]
        row = dict(dataset=d, name=config.DATASETS[d], steps=len(steps), best_label=k.max(),
                   median_predicted=steps["predicted_improvement"].median() if len(steps) else np.nan)
        if len(steps):
            first = steps.iloc[0]
            row.update(feature=first["feature"], predicted=first["predicted_improvement"],
                       true=k.get(first["feature"], np.nan),
                       percentile=(k < k.get(first["feature"], np.nan)).mean() if first["feature"] in k else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------------------- figures

def fig1_label_reliability(numbers):
    """Two independent measurements of the same candidates' label: AutoGluon (one fit per seed) vs repeated CV."""
    cand = pd.read_parquet(REL / "candidates" / "candidates.parquet")
    cand = cand[cand["in_random_sample"] & ~cand["feature - name"].str.startswith("without - ")]
    ag = pd.concat([pd.read_parquet(p) for p in sorted((REL / "ag").glob("*.parquet"))])
    ag = ag.merge(cand[["dataset - id", "feature - name"]], on=["dataset - id", "feature - name"])
    A = ag.pivot_table(index=["dataset - id", "feature - name"], columns="seed", values="label").dropna()
    A = A[[1, 2, 3]]
    per_ag = [dict(dataset=d, spearman=spearmanr(g[1], g[2]).correlation, icc=icc1(g.to_numpy()), n=len(g))
              for d, g in A.groupby(level=0) if len(g) >= 20]
    per_ag = pd.DataFrame(per_ag)
    A = A.loc[per_ag["dataset"].tolist()]
    ag_x, ag_y = pct_rank(A[1], A.index.get_level_values(0)), pct_rank(A[2], A.index.get_level_values(0))

    rows, pts = [], []
    for p in sorted((KB / "label_repeats" / "GBM").glob("*.parquet")):
        r = pd.read_parquet(p)
        r = r[~r["feature - name"].str.startswith("without - ")]
        M = r[[f"repeat_{i}" for i in range(config.CV_REPEATS)]].to_numpy()
        a, b = M[:, :5].mean(1), M[:, 5:].mean(1)
        rows.append(dict(dataset=int(p.stem), split_half=spearmanr(a, b).correlation,
                         icc10=spearman_brown(icc1(M), config.CV_REPEATS)))
        pts.append(pd.DataFrame({"x": pd.Series(a).rank(pct=True), "y": pd.Series(b).rank(pct=True)}))
    per_cv = pd.DataFrame(rows)
    pts = pd.concat(pts).sample(4000, random_state=0)
    numbers["label_reliability"] = dict(
        ag_datasets=len(per_ag), ag_seed_spearman_median=per_ag["spearman"].median(),
        ag_icc_single_fit_median=per_ag["icc"].median(), cv_datasets=len(per_cv),
        cv_split_half_spearman_median=per_cv["split_half"].median(), cv_icc10_median=per_cv["icc10"].median(),
        cv_icc10_ge_0_5=int((per_cv["icc10"] >= 0.5).sum()))

    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.2), sharey=True)
    for ax, (x, y), color, title, text in [
            (axes[0], (ag_x, ag_y), ORANGE, "(a) AutoGluon label: seed 1 vs. seed 2",
             f"Spearman {per_ag['spearman'].median():.2f}\nICC {per_ag['icc'].median():.2f}"),
            (axes[1], (pts["x"], pts["y"]), BLUE, "(b) CV label: repeats 1-5 vs. 6-10",
             f"Spearman {per_cv['split_half'].median():.2f}\nICC (10 repeats) {per_cv['icc10'].median():.2f}")]:
        ax.scatter(x, y, s=3, color=color, alpha=0.35, linewidths=0, rasterized=True)
        ax.plot([0, 1], [0, 1], color=NEUTRAL, lw=0.8, ls="--")
        ax.set_title(title, loc="left")
        ax.text(0.03, 0.97, text + "\n(median over datasets)", transform=ax.transAxes, va="top", color=INK,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))
        ax.set_xlabel("label rank, measurement 1")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
    axes[0].set_ylabel("label rank, measurement 2")
    save(fig, "fig1_label_reliability")


def fig2_fe_comparison(gains, numbers):
    test = gains["test"]
    methods = ["TabPrep", "OpenFE"] + [f"MetaFE_{t}" for t in TIME_LIMITS]
    labels = ["TabPrep", "OpenFE"] + [f"MetaFE\n{t} s" for t in TIME_LIMITS]
    colors = [BLUE, AQUA] + [ORANGE] * len(TIME_LIMITS)
    rows = []
    for m in methods:
        g = test[m].dropna()
        rows.append(dict(method=m, datasets=len(g), median=g.median(), mean=g.mean(), better=int((g > 0.05).sum()),
                         worse=int((g < -0.05).sum()), wilcoxon_p=wilcoxon(g).pvalue))
    table = pd.DataFrame(rows)
    numbers["fe_comparison"] = table.set_index("method").to_dict("index")
    lines = ["\\begin{table}[t]", "\\centering",
             "\\caption{Test error decrease over the original data (\\%, LightGBM, mean over 3 seeds); better/worse: "
             "datasets with a change $>0.05$\\,\\%; $p$: two-sided Wilcoxon signed-rank test.}", "\\label{tab:fe}",
             "\\begin{tabular}{lrrrrrr}", "\\hline",
             "Method & Datasets & Median & Mean & Better & Worse & $p$ \\\\", "\\hline"]
    for r in rows:
        name = r["method"].replace("MetaFE_", "MetaFE ") + (" s" if r["method"].startswith("MetaFE") else "")
        lines.append(f"{name} & {r['datasets']} & {r['median']:+.2f} & {r['mean']:+.2f} & {r['better']} & "
                     f"{r['worse']} & {r['wilcoxon_p']:.3f} \\\\")
    lines += ["\\hline", "\\end{tabular}", "\\end{table}"]
    (OUT / "table_fe_comparison.tex").write_text("\n".join(lines) + "\n")

    fig, ax = plt.subplots(figsize=(FULL_W, 2.3))
    rng = np.random.RandomState(0)
    for i, (m, c) in enumerate(zip(methods, colors)):
        g = test[m].dropna().to_numpy()
        ax.boxplot(g, positions=[i], widths=0.5, showfliers=False, medianprops=dict(color=INK, lw=1.2),
                   boxprops=dict(color=INK2, lw=0.8), whiskerprops=dict(color=INK2, lw=0.8),
                   capprops=dict(color=INK2, lw=0.8))
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(g)), g, s=9, color=c, alpha=0.8, edgecolors="white",
                   linewidths=0.4, zorder=3)
        ax.text(i, 210, f"{np.median(g):+.1f} %", ha="center", va="bottom", color=INK, fontsize=6.5)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_yscale("symlog", linthresh=5)
    ax.set_yticks([-200, -50, -10, -5, 0, 5, 10, 50, 100])
    ax.set_yticklabels(["-200", "-50", "-10", "-5", "0", "5", "10", "50", "100"])
    ax.set_ylim(-320, 330)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("test error decrease vs. original (%)")
    ax.grid(axis="x", visible=False)
    save(fig, "fig2_fe_comparison")


def fig3_metafe_picks(picks, numbers):
    p = picks.dropna(subset=["true"])
    numbers["metafe_first_pick"] = dict(
        datasets=len(p), datasets_without_step=int((picks["steps"] == 0).sum()),
        predicted_median_pct=100 * p["predicted"].median(), true_median_pct=100 * p["true"].median(),
        true_positive=int((p["true"] > 0).sum()), percentile_median=p["percentile"].median(),
        best_candidate_median_pct=100 * picks["best_label"].median(),
        median_steps=picks["steps"].median(), median_predicted_step_pct=100 * picks["median_predicted"].median())

    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.1), gridspec_kw=dict(width_ratios=[1.15, 1]))
    ax = axes[0]
    ax.scatter(100 * p["predicted"], 100 * p["true"], s=12, color=ORANGE, edgecolors="white", linewidths=0.4,
               zorder=3)
    ax.plot([0.8, 30], [0.8, 30], color=NEUTRAL, lw=0.8, ls="--")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10, 20])
    ax.set_xticklabels(["1", "2", "5", "10", "20"])
    ax.set_xlim(0.8, 30)
    ax.set_yscale("symlog", linthresh=1)
    ax.set_yticks([-5, -1, 0, 1, 5])
    ax.set_yticklabels(["-5", "-1", "0", "1", "5"])
    ax.set_xlabel("predicted improvement (%)")
    ax.set_ylabel("true improvement (CV label, %)")
    ax.set_title("(a) MetaFE's first feature per dataset", loc="left")
    ax.text(6, 9, "y = x", ha="right", color=NEUTRAL, fontsize=6)
    ax = axes[1]
    ax.hist(100 * p["percentile"], bins=np.arange(0, 101, 10), color=ORANGE, edgecolor="white", linewidth=1)
    ax.axhline(len(p) / 10, color=INK2, lw=0.8, ls="--")
    ax.text(2, len(p) / 10 + 0.2, "random pick", color=INK2, fontsize=6, va="bottom")
    ax.set_xlabel("percentile among the candidates")
    ax.set_ylabel("datasets")
    ax.set_xlim(0, 100)
    ax.set_title("(b) rank of the pick by its true label", loc="left")
    ax.grid(axis="x", visible=False)
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig3_metafe_picks")


def fig4_transfer_gap(numbers):
    old = pd.read_csv(CHECK_OLD / "step1_lodo" / "per_dataset.csv")
    new = pd.read_csv(CHECK / "step1_lodo" / "per_dataset.csv")
    within = pd.read_csv(CHECK / "within_dataset" / "per_dataset.csv")
    old, new = old[old["subset"] == "add"], new[new["subset"] == "add"]
    rows = [  # (label, per-dataset Spearman, color)
        ("describe() MF, AutoGluon labels", old[old["ranker"] == "sur_old"]["spearman"], NEUTRAL),
        ("target-aware MF, AutoGluon labels", old[old["ranker"] == "sur_new_rank"]["spearman"], NEUTRAL),
        ("describe() MF, CV labels", new[new["ranker"] == "sur_old"]["spearman"], BLUE),
        ("residual heuristic (no surrogate)", new[new["ranker"] == "heu_eta_res"]["spearman"], YELLOW),
        ("target-aware MF, CV labels", new[new["ranker"] == "sur_new_rank"]["spearman"], BLUE),
        ("same dataset: target-aware MF, CV labels", within[within["ranker"] == "within_operand"]["spearman"], AQUA),
        ("label-noise ceiling", within.groupby("dataset")["ceiling_spearman"].first(), INK2)]
    numbers["spearman_medians"] = {lab: float(np.median(v)) for lab, v, _ in rows}
    top = new[new["ranker"] == "sur_new_rank"]
    numbers["top10_excess_pct"] = dict(
        lodo_target_aware=100 * top["top10_excess"].median(),
        residual_heuristic=100 * new[new["ranker"] == "heu_eta_res"]["top10_excess"].median(),
        same_dataset=100 * within[within["ranker"] == "within_operand"]["top10_excess"].median(),
        oracle=100 * top["oracle_top10_excess"].median())

    curve = pd.read_csv(CHECK / "learning_curve" / "per_dataset.csv")
    by_ds = curve.groupby(["ranker", "n", "dataset"])["spearman"].mean().reset_index()
    numbers["learning_curve_median"] = {f"{r}_{n}": float(g["spearman"].median())
                                        for (r, n), g in by_ds.groupby(["ranker", "n"])}

    fig, ax = plt.subplots(figsize=(FULL_W, 2.3))
    rng = np.random.RandomState(0)
    for i, (lab, v, c) in enumerate(rows):
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        y = len(rows) - 1 - i
        ax.scatter(v, y + rng.uniform(-0.17, 0.17, len(v)), s=5, color=c, alpha=0.7, linewidths=0)
        ax.plot([np.median(v)] * 2, [y - 0.32, y + 0.32], color=INK, lw=1.6)
        ax.text(1.02, y, f"{np.median(v):.2f}", va="center", fontsize=6.5, color=INK)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1])
    ax.axvline(0, color=INK2, lw=0.8)
    ax.axhspan(1.5, len(rows) - 0.5, color=GRID, alpha=0.35, linewidth=0, zorder=0)
    ax.text(0.98, len(rows) - 0.85, "trained on the 43 other datasets", fontsize=6, color=INK2, va="center",
            ha="right")
    ax.set_xlim(-0.5, 1.0)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel("Spearman of predicted and true label, per held-out dataset")
    ax.grid(axis="y", visible=False)
    save(fig, "fig4_transfer_gap")
    return by_ds, float(np.median(rows[-1][1]))


def fig5_learning_curve(by_ds, ceil):
    fig, ax = plt.subplots(figsize=(HALF_W * 1.3, 2.1))
    for r, c, lab in [("local", AQUA, "labels of the dataset only"), ("local+kb", VIOLET, "+ KB surrogate as input")]:
        g = by_ds[by_ds["ranker"] == r].groupby("n")["spearman"]
        med, q1, q3 = g.median(), g.quantile(0.25), g.quantile(0.75)
        ax.plot(med.index, med.values, color=c, marker="o", ms=3, label=lab)
        if r == "local":
            ax.fill_between(med.index, q1.values, q3.values, color=c, alpha=0.15, linewidth=0)
    kb0 = by_ds[by_ds["ranker"] == "kb"]["spearman"].median()
    ax.axhline(kb0, color=BLUE, lw=1, ls="--")
    ax.text(10, kb0 + 0.02, "KB of 43 other datasets", color=BLUE, fontsize=6, va="bottom")
    ax.axhline(ceil, color=INK2, lw=1, ls=":")
    ax.text(10, ceil + 0.02, "label-noise ceiling", color=INK2, fontsize=6, va="bottom")
    ax.set_xscale("log")
    ax.set_xticks([10, 25, 50, 100, 200, 400])
    ax.set_xticklabels(["10", "25", "50", "100", "200", "400"])
    ax.set_ylim(0, 1)
    ax.set_xlabel("labelled candidates of the new dataset")
    ax.set_ylabel("median Spearman")
    ax.legend(loc="lower right")
    save(fig, "fig5_learning_curve")


def fig6_dataset_selection(numbers):
    summary = pd.read_csv(CHECK / "dataset_selection" / "summary.csv")
    numbers["dataset_selection"] = summary.set_index(["setting", "selector"]).to_dict("index")
    numbers["dataset_selection"] = {f"{k[0]}|{k[1]}": v for k, v in numbers["dataset_selection"].items()}
    names = {"always_Original": "always original", "always_TabPrep": "always TabPrep", "always_OpenFE": "always OpenFE",
             "meta": "meta-learned selector", "validation": "best validation score", "oracle": "oracle (test)"}
    colors = {"always_Original": NEUTRAL, "always_TabPrep": BLUE, "always_OpenFE": AQUA, "meta": ORANGE,
              "validation": YELLOW, "oracle": INK2}
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 1.9), sharex=True)
    for ax, (setting, title) in zip(axes, [("TabPrep", "(a) original or TabPrep (44 datasets)"),
                                           ("TabPrep+OpenFE", "(b) original, TabPrep or OpenFE (36)")]):
        s = summary[summary["setting"] == setting].reset_index(drop=True)
        for i, r in s.iterrows():
            y = len(s) - 1 - i
            ax.plot([r["median_gain"], r["mean_gain"]], [y, y], color=GRID, lw=2, zorder=1)
            ax.scatter(r["mean_gain"], y, s=22, color=colors[r["selector"]], zorder=3)
            ax.scatter(r["median_gain"], y, s=22, facecolors="white", edgecolors=colors[r["selector"]], lw=1.2,
                       zorder=3)
            if r["selector"] == "meta":
                ax.text(max(r["mean_gain"], r["median_gain"]) + 1.5, y, f"p = {r['perm_p']:.2f}", va="center",
                        fontsize=6, color=INK2)
        ax.set_yticks(range(len(s)))
        ax.set_yticklabels([names[x] for x in s["selector"]][::-1])
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_title(title, loc="left")
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("test error decrease (%)")
    axes[0].scatter([], [], s=22, color=INK2, label="mean")
    axes[0].scatter([], [], s=22, facecolors="white", edgecolors=INK2, label="median")
    fig.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.07), handletextpad=0.2)
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig6_dataset_selection")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    numbers = {}
    kb = kb_labels()
    numbers["knowledge_base"] = dict(datasets=int(kb["dataset - id"].nunique()), labels=len(kb))
    gains = fe_gains()
    fig1_label_reliability(numbers)
    fig2_fe_comparison(gains, numbers)
    picks = metafe_first_picks(kb)
    picks.to_csv(OUT / "metafe_first_picks.csv", index=False)
    fig3_metafe_picks(picks, numbers)
    fig5_learning_curve(*fig4_transfer_gap(numbers))
    fig6_dataset_selection(numbers)
    (OUT / "numbers.json").write_text(json.dumps(numbers, indent=2, default=float))
    print(json.dumps(numbers, indent=2, default=float))


if __name__ == "__main__":
    main()

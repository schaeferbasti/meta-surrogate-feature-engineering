"""Plots and LaTeX tables of the evaluation (fold-averaged Result_<id>.parquet files of merge_folds).

Methods: Original, OpenFE, TabPrep and MetaFE at every time limit ("MetaFE (1000)"). Every model gets its own folder
results/<experiment>/plots/<model>/; with several models, plots/average/ shows the improvement over Original averaged
over the models (per dataset; MetaFE for each model is the MetaFE data for that model). Two method sets are plotted:
"all" (every method) and "comparison" (MetaFE at --metafe_time_limit vs. OpenFE, TabPrep and Original).
Errors are -score (log loss / RMSE), improvements are relative to Original in percent (positive = lower error).
summary.csv in every folder: mean / median improvement and number of datasets better than Original per method.

    python -m metafe.evaluation.plots --experiment lightgbm [--metafe_time_limit 1000]
"""
import argparse
import glob
import os
import re
from math import ceil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from metafe import config

BASELINE = "Original"
COLORS = {"OpenFE": "#fed9d9", "TabPrep": "#ffb36d", "MetaFE": "#6db1ff"}
OPENFE_RUNTIME_S = 269.05  # average OpenFE runtime per dataset, measured in the first experiments


def display_name(origin):
    match = re.fullmatch(r"MetaFE_(\d+)", origin)
    return f"MetaFE ({match.group(1)})" if match else origin


def load_results(experiment):
    """All fold-averaged results: origin (display name), dataset (name), model (without _BAG_L1), scores."""
    files = [f for f in glob.glob(os.path.join(experiment.evaluation_dir, "Result_*.parquet"))
             if re.fullmatch(r"Result_\d+\.parquet", os.path.basename(f))]
    if not files:
        raise SystemExit(f"No Result_<id>.parquet in {experiment.evaluation_dir}, run metafe.evaluation.merge_folds")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["origin"] = df["origin"].map(display_name)
    df["dataset"] = df["dataset"].astype(int).map(lambda d: config.DATASETS.get(d, str(d)))
    df["model"] = df["model"].str.replace("_BAG_L1", "", regex=False)
    return df


def error_tables(df):
    """Errors and their std per dataset (rows) and method (columns), for val and test, of one model."""
    tables = {}
    for split in ["val", "test"]:
        tables[split] = -df.pivot(index="dataset", columns="origin", values=f"score_{split}_mean").sort_index()
        tables[f"{split}_std"] = df.pivot(index="dataset", columns="origin", values=f"score_{split}_std").sort_index()
    return tables


def improvement(errors, std=None):
    """Relative improvement over Original (%) of every other method (and the std scaled the same way)."""
    baseline = errors[BASELINE]
    methods = [c for c in errors.columns if c != BASELINE]
    impr = errors[methods].apply(lambda col: 100 * (baseline - col) / baseline).replace([np.inf, -np.inf], np.nan)
    if std is None:
        return impr
    return impr, std[methods].apply(lambda col: 100 * col / baseline)


def ordered_methods(columns):
    # Original, MetaFE by time limit, then the baselines
    metafe = sorted([c for c in columns if c.startswith("MetaFE")], key=lambda c: int(re.sub(r"\D", "", c)))
    return [c for c in [BASELINE] if c in columns] + metafe + [c for c in ["OpenFE", "TabPrep"] if c in columns]


def colors_for(methods):
    cmap = plt.get_cmap("nipy_spectral")
    spectral = [cmap(x) for x in np.linspace(0, 0.95, len(methods))]
    return [COLORS.get(m.split(" ")[0], spectral[i]) if len(methods) <= 4 else spectral[i] for i, m in enumerate(methods)]


def save(fig, out_dir, name):
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, name + ".pdf"), bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_errors(errors, out_dir, name, split):
    fig, ax = plt.subplots(figsize=(12, 8))
    for method, color in zip(errors.columns, colors_for(list(errors.columns))):
        ax.plot(errors.index, errors[method], marker="o", linestyle="None", label=method, color=color)
    ax.set_yscale("log")
    ax.set_xlabel("Dataset")
    ax.set_ylabel(f"{split.title()} error")
    ax.set_title(f"{split.title()} error of the model on the feature-engineered datasets", fontsize=16)
    ax.tick_params(axis="x", rotation=90, labelsize=16)
    ax.tick_params(axis="y", labelsize=16)
    ax.legend(fontsize=16)
    ax.grid(True)
    save(fig, out_dir, f"Graph_{name}")


def plot_improvement_per_dataset(impr, impr_std, out_dir, name):
    fig, ax = plt.subplots(figsize=(9, 5))
    impr.plot.bar(ax=ax, color=colors_for(list(impr.columns)), width=0.8, yerr=impr_std, capsize=3,
                  error_kw={"elinewidth": 1, "alpha": 0.8})
    ax.set_yscale("symlog")
    ax.set_ylabel("Improvement over Original (%)", fontsize=16)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=90, labelsize=16)
    ax.tick_params(axis="y", labelsize=16)
    ax.legend(fontsize=16)
    ax.grid(True, axis="y", alpha=0.3)
    save(fig, out_dir, f"Graph_Improvement_{name}")


def plot_count_best(impr_val, impr_test, out_dir, name):
    fig, ax = plt.subplots(figsize=(12, 8))
    impr_val.dropna(how="all").idxmax(axis=1).value_counts().plot(
        kind="bar", ax=ax, color="#cfe4ff", label="Number of datasets with the lowest validation error")
    impr_test.dropna(how="all").idxmax(axis=1).value_counts().plot(
        kind="bar", ax=ax, width=0.3, color="#6db1ff", label="Number of datasets with the lowest test error")
    ax.set_ylabel("Number of datasets", fontsize=16)
    ax.tick_params(axis="x", rotation=90, labelsize=16)
    ax.legend(fontsize=16)
    ax.grid(True)
    save(fig, out_dir, f"Count_Best_{name}")


def plot_average_improvement(impr, out_dir, name):
    average = impr.mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(max(3, 0.6 * len(average)), 5))
    average.plot(kind="bar", ax=ax, color="#6db1ff")
    for i, value in enumerate(average):
        ax.text(i, -0.1 if value >= 0 else 0, f"{value:.2f}%", ha="center", va="top" if value >= 0 else "bottom",
                fontsize=8)
    ax.set_yscale("symlog", linthresh=1)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Avg. Improvement over Original (%)", fontsize=16)
    ax.tick_params(axis="x", rotation=90, labelsize=16)
    ax.grid(True, linestyle="--", alpha=0.6)
    save(fig, out_dir, f"Average_Percentage_Improvement_{name}")


def plot_boxplot_improvement(impr, out_dir, name):
    impr = impr[impr.mean().sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(max(4, 0.6 * impr.shape[1]), 5))
    bp = impr.boxplot(ax=ax, grid=True, return_type="dict")
    for part, color, width in [("boxes", "black", 1.5), ("medians", "#e81814", 2), ("whiskers", "#333333", 1.5),
                               ("caps", "#333333", 1.5)]:
        for line in bp[part]:
            line.set(color=color, linewidth=width)
    rng = np.random.default_rng(0)
    for i, method in enumerate(impr.columns):
        y = impr[method].dropna()
        ax.plot(rng.normal(i + 1, 0.05, len(y)), y, "o", alpha=0.4, markersize=4, color="#6db1ff")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_yscale("symlog", linthresh=1)
    ax.set_ylabel("Improvement over Original (%)", fontsize=16)
    ax.tick_params(axis="x", rotation=90, labelsize=16)
    save(fig, out_dir, f"Boxplot_Percentage_Improvement_{name}")


def metafe_runtimes(log_dirs, time_limits):
    """Average runtime per dataset of MetaFE with each time limit, from the step logs (a run ends at its last step)."""
    ends = []
    for path in [p for d in log_dirs for p in glob.glob(os.path.join(d, "FE_*_MetaFE_log.csv"))]:
        log = pd.read_csv(path)
        finished = log[log["step"].astype(str) == "end"]
        ends.append(finished["elapsed_s"].iloc[0] if len(finished) else np.inf)  # no end row: killed at the limit
    if not ends:
        return {}
    return {f"MetaFE ({t})": float(np.mean(np.minimum(ends, t))) for t in time_limits}


def plot_pareto(impr, runtimes, out_dir):
    from adjustText import adjust_text
    from matplotlib.lines import Line2D

    runtimes = {m: t for m, t in runtimes.items() if m in impr.columns}
    if not runtimes:
        print("No runtimes, Pareto plot skipped")
        return
    performance = impr.fillna(0).mean()
    points = pd.DataFrame({"Method": list(runtimes), "Runtime": list(runtimes.values()),
                           "Performance": [performance[m] for m in runtimes]})
    points["Pareto"] = [not ((points["Performance"] > p) & (points["Runtime"] < t)).any()
                        for p, t in zip(points["Performance"], points["Runtime"])]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(points["Runtime"], points["Performance"], s=100, alpha=0.6, zorder=3,
               c=["#e81814" if p else "black" for p in points["Pareto"]])
    front = points[points["Pareto"]].sort_values("Runtime")
    if len(front) > 1:
        ax.plot(front["Runtime"], front["Performance"], color="#e81814", linestyle="--", linewidth=2, zorder=2)
    texts = [ax.text(r, p, m, fontsize=14, ha="center", va="top", zorder=4, color="grey")
             for m, r, p in zip(points["Method"], points["Runtime"], points["Performance"])]
    adjust_text(texts, ax=ax, x=points["Runtime"].to_numpy(), y=points["Performance"].to_numpy(),
                force_text=(0, -5.0), force_static=(0.0, 0.0), only_move={"text": "x", "static": "x",
                "explode": "x", "pull": "x"}, arrowprops=dict(arrowstyle="-", color="gray", lw=0.5, alpha=0.45))
    ax.margins(x=0.2, y=0.2)
    ax.set_xscale("log")
    ax.set_xlabel("Average Runtime per Dataset (s)", fontsize=16)
    ax.set_ylabel("Average Improvement over Original (%)", fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.legend(handles=[
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#e81814", markersize=10, label="Pareto Efficient"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markersize=10, alpha=0.6, label="Dominated"),
        Line2D([0], [0], color="#e81814", linestyle="--", linewidth=2, label="Pareto Front")], fontsize=14)
    save(fig, out_dir, "Pareto_MetaFE_OpenFE")


def latex_table(errors, std, model, columns_per_table=5):
    """Test errors with std, split into tables of columns_per_table methods."""
    caption = f"Test error of {model} on the feature-engineered datasets"
    tables = []
    methods = list(errors.columns)
    for i in range(ceil(len(methods) / columns_per_table)):
        columns = methods[i * columns_per_table:(i + 1) * columns_per_table]
        lines = [r"\begin{table}[h!]", r"    \footnotesize",
                 rf"    \begin{{tabular*}}{{\textwidth}}{{@{{\extracolsep{{0.2em}}}} c|{'c' * len(columns)} @{{}}}}",
                 r"        \toprule",
                 "        " + " & ".join(["Dataset"] + [f"\\makecell{{{c.replace(' ', chr(92) * 2)}}}" for c in columns])
                 + r" \\", r"        \midrule"]
        for dataset in errors.index:
            cells = [dataset]
            for c in columns:
                value, s = errors.loc[dataset, c], std.loc[dataset, c]
                cells.append("/" if pd.isna(value) else f"${value:.2f}$" if pd.isna(s)
                             else f"${value:.2f} {{\\scriptscriptstyle \\pm {s:.2f}}}$")
            lines.append("        " + " & ".join(cells) + r" \\")
        lines += [r"        \bottomrule", r"    \end{tabular*}"]
        lines += ([rf"    \caption{{{caption}}}", rf"    \label{{tab:test_{model}}}"] if i == 0
                  else [r"    \ContinuedFloat", rf"    \caption*{{{caption} (cont.)}}"])
        lines += [r"\end{table}", ""]
        tables.append("\n".join(lines))
    return "\n".join(tables)


def summary(impr):
    """Mean / median improvement over Original and number of datasets better than Original per method."""
    return pd.DataFrame({"mean_improvement_%": impr.mean(), "median_improvement_%": impr.median(),
                         "better": (impr > 0).sum(), "datasets": impr.notna().sum()}).sort_values(
        "mean_improvement_%", ascending=False)


def average_over_models(views, split):
    """Improvement per dataset averaged over the models. A method gets a value for a dataset only if it has a result
    for every model that has an Original result for the dataset (models that fail on a dataset are left out)."""
    stacked = pd.concat({model: v[split] for model, v in views.items()}, names=["model"])
    n_models = sum(v["errors"][split][BASELINE].notna().astype(int) for v in views.values())
    per_dataset = stacked.groupby(level="dataset")
    return per_dataset.mean().where(per_dataset.count().eq(n_models.reindex(per_dataset.count().index), axis=0))


def plot_view(view, out_dir, metafe_time_limit, runtimes):
    """All plots of one view: {"val"/"test": improvement table, "*_std": its std or None, "errors": error tables of a
    single model or None}."""
    os.makedirs(out_dir, exist_ok=True)
    methods = ordered_methods(list(view["test"].columns) + [BASELINE])[1:]
    main_metafe = f"MetaFE ({metafe_time_limit})"
    comparison = [m for m in [main_metafe, "OpenFE", "TabPrep"] if m in methods]
    for subset_name, subset in [("all", methods), ("comparison", comparison)]:
        rename = {main_metafe: "MetaFE"} if subset_name == "comparison" else {}
        impr = {k: view[k][subset].rename(columns=rename) for k in ["val", "test"]}
        impr_std = {k: None if view[f"{k}_std"] is None else view[f"{k}_std"][subset].rename(columns=rename)
                    for k in ["val", "test"]}
        for split in ["val", "test"]:
            name = f"{split.title()}_{subset_name}"
            if view["errors"] is not None:
                errors = view["errors"][split][[BASELINE] + subset].rename(columns=rename)
                plot_errors(errors, out_dir, name, "validation" if split == "val" else "test")
            plot_improvement_per_dataset(impr[split], impr_std[split], out_dir, name)
            plot_average_improvement(impr[split], out_dir, name)
            plot_boxplot_improvement(impr[split], out_dir, name)
        plot_count_best(impr["val"], impr["test"], out_dir, subset_name)
    plot_pareto(view["test"][methods], runtimes, out_dir)
    table = summary(view["test"][methods])
    table.to_csv(os.path.join(out_dir, "summary.csv"))
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    config.add_experiment_argument(parser)
    parser.add_argument("--metafe_time_limit", type=int, default=1000, help="MetaFE run of the 'comparison' plots")
    args = parser.parse_args()
    experiment = config.EXPERIMENTS[args.experiment]
    from metafe.models import MODEL_NAMES

    df = load_results(experiment)
    key_of = {name: key for key, name in MODEL_NAMES.items()}
    views, log_dirs = {}, {}
    for model, model_df in df.groupby("model"):
        errors = error_tables(model_df)
        val, val_std = improvement(errors["val"], errors["val_std"])
        test, test_std = improvement(errors["test"], errors["test_std"])
        views[model] = {"val": val, "val_std": val_std, "test": test, "test_std": test_std, "errors": errors}
        log_dirs[model] = [experiment.metafe_dir(key_of.get(model, model))]
        out_dir = os.path.join(experiment.plots_dir, model)
        os.makedirs(out_dir, exist_ok=True)
        methods = ordered_methods(errors["test"].columns)
        with open(os.path.join(out_dir, "test_errors.tex"), "w") as f:
            f.write(latex_table(errors["test"][methods], errors["test_std"][methods], model))
    if len(views) > 1:
        views["average"] = {k: average_over_models(views, k) for k in ["val", "test"]}
        views["average"].update({"val_std": None, "test_std": None, "errors": None})
        log_dirs["average"] = [d for dirs in log_dirs.values() for d in dirs]

    for name, view in views.items():
        runtimes = metafe_runtimes(log_dirs[name], experiment.time_limits)
        runtimes["OpenFE"] = OPENFE_RUNTIME_S
        table = plot_view(view, os.path.join(experiment.plots_dir, name), args.metafe_time_limit, runtimes)
        print(f"=== {name}: test improvement over Original (%)\n{table.round(2).to_string()}")
    print(f"Plots, summary.csv and test_errors.tex written to {experiment.plots_dir}/<model or average>")


if __name__ == "__main__":
    main()

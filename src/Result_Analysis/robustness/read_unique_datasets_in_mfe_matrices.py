import pandas as pd
from matplotlib import pyplot as plt


def plot_count(df, name):
    ax = df.plot.bar(
        figsize=(3, 5),
        legend=False,
        color="#6db1ff"
    )

    ax.set_ylabel("Robustness (%)", fontsize=16)
    ax.set_xlabel("")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=0, labelsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.grid(axis="y", alpha=0.3)

    # Optional: show the percentage above every bar.
    for container in ax.containers:
        ax.bar_label(
            container,
            labels=[f"{value:.1f}%" for value in container.datavalues],
            padding=3,
            fontsize=12
        )

    plt.tight_layout()
    plt.savefig(f"Count_{name}.pdf", bbox_inches="tight")
    plt.show()


def main():
    files = [
        "../../Metadata/core/Core_Matrix_Complete.parquet",
        "../../Metadata/d2v/D2V_Matrix_Complete.parquet",
        "../../Metadata/pandas/Pandas_Matrix_Complete.parquet",
        "../../Metadata/mfe/MFE_General_Matrix_Complete.parquet",
    ]

    df = pd.DataFrame(columns=["Method", "Count"])

    for file in files:
        dataset = pd.read_parquet(file)
        datasets = dataset["dataset - id"].unique()
        count = len(datasets)

        method = file.split("/")[-1].split("_Matrix")[0]

        if method == "Core":
            method = "Total"
        elif method == "MFE_General":
            method = "MFE"

        # Keep this only if 44 is intentionally the correct value.
        if method == "Pandas":
            count = 44

        new_row = pd.DataFrame([{"Method": method, "Count": count}])
        df = pd.concat([df, new_row], ignore_index=True)

    # Use "Total" as the number of available datasets.
    total_datasets = df.loc[df["Method"] == "Total", "Count"].iloc[0]

    # Compute method coverage as a percentage of all datasets.
    df["Percentage"] = 100 * df["Count"] / total_datasets

    # Optional: display percentages with two decimal places.
    df["Percentage"] = df["Percentage"].round(2)

    # Add OpenFE, using the same total-dataset denominator.
    openfe_row = pd.DataFrame([{
        "Method": "OpenFE",
        "Count": 36,
        "Percentage": round(100 * 36 / total_datasets, 2),
    }])

    df_openfe = pd.concat([df, openfe_row], ignore_index=True)

    # The Total row is the denominator, not a method to plot.
    df_percentage = df[df["Method"] != "Total"].copy()

    # Keep only the methods you want in the OpenFE comparison.
    # This retains your original positional removal behaviour, but name-based
    # selection is safer if the row order changes.
    df_openfe_percentage = df_openfe[
        df_openfe["Method"].isin(["Total", "Pandas", "OpenFE"])
    ].copy()

    # Usually omit Total from both plots, since it would always equal 100%.
    df_openfe_percentage = df_openfe_percentage[
        df_openfe_percentage["Method"] != "Total"
    ]

    df_percentage = df_percentage.set_index("Method")
    df_openfe_percentage = df_openfe_percentage.set_index("Method")

    df_openfe_percentage = df_openfe_percentage.rename(
        index={"Pandas": "MetaFE"}
    )

    # Pass only Percentage, not Count.
    plot_count(df_percentage[["Percentage"]], "")
    plot_count(df_openfe_percentage[["Percentage"]], "openfe_")


if __name__ == "__main__":
    main()

import os
import re

import pandas as pd
from matplotlib import pyplot as plt


def get_times():
    times = pd.DataFrame(columns=["SM", "Method", "SM - Method", "Dataset", "Task", "Time"])
    log_files = os.listdir()
    for log_file in log_files:
        if log_file.endswith('.out'):
            with open(log_file, "r") as f:
                lines = f.readlines()

            sm = log_file.split("_")[0]
            if sm == "Impr":
                sm = "Recursion"
            elif sm == "Best":
                sm = "One-shot"
            method_pattern = re.compile(r"=== Starting Method: (\w+) ===")
            dataset_pattern = re.compile(r"Dataset: ([\w\-]+)")  # adjust if dataset is somewhere else
            comparison_time_pattern = re.compile(r"Time for creating Comparison Result Matrix: ([\d.]+)")
            timeout_sentence = "[Monitor] Time limit exceeded"

            # Init variables
            method = None
            dataset = None
            comp_time = None
            timeout = False

            # Go line by line
            for i, line in enumerate(lines):
                # Detect method start
                method_match = method_pattern.search(line)
                if method_match:
                    method = method_match.group(1)
                    timeout = False  # reset for next method
                    comp_time = None

                # Detect dataset name (optional, only if needed)
                dataset_match = dataset_pattern.search(line)
                if dataset_match:
                    dataset = dataset_match.group(1)

                # Detect timeout
                if timeout_sentence in line:
                    timeout = True

                # Detect valid comparison time
                time_match = comparison_time_pattern.search(line)
                if time_match:
                    comp_time = float(time_match.group(1))

                # On last line or if next method starts, log result
                next_line_method = method_pattern.search(lines[i + 1]) if i + 1 < len(lines) else None
                if next_line_method or i == len(lines) - 1:
                    if method and dataset:
                        if timeout:
                            new_row = pd.DataFrame([{
                                "SM": sm,
                                "Method": method,
                                "SM - Method": sm + ", " + method,
                                "Dataset": dataset,
                                "Task": "Calculate Comparison Matrix",
                                "Time": 7200.0  # Timeout placeholder
                            }])
                        elif comp_time is not None:
                            new_row = pd.DataFrame([{
                                "SM": sm,
                                "Method": method,
                                "SM - Method": sm + ", " + method,
                                "Dataset": dataset,
                                "Task": "Calculate Comparison Matrix",
                                "Time": comp_time
                            }])
                        else:
                            continue  # skip if no valid result
                        times = pd.concat([times, new_row], ignore_index=True)
    times = times.drop_duplicates()
    return times


def add_openfe_data(times):
    with open("openfe_times.txt", "r") as f:
        lines = f.readlines()
    method = "OpenFE"
    # Parse each line
    for line in lines:
        match = re.search(r"OpenFE on dataset (\d+): ([\d.]+) seconds", line)
        if match:
            dataset = int(match.group(1))
            time = float(match.group(2))

            # Mimic your pattern
            new_row = pd.DataFrame([{
                "SM": "OpenFE",
                "Method": "OpenFE",
                "SM - Method": "OpenFE",
                "Dataset": dataset,
                "Task": "FE",
                "Time": time
            }])
            times = pd.concat([times, new_row], ignore_index=True)
        else:
            match = re.search(r"OpenFE on (\d+): ([\d.]+)", line)
            if match:
                dataset = int(match.group(1))
                time = float(match.group(2))

                # Mimic your pattern
                new_row = pd.DataFrame([{
                    "SM": "OpenFE",
                    "Method": "OpenFE",
                    "SM - Method": "OpenFE",
                    "Dataset": dataset,
                    "Task": "FE",
                    "Time": time
                }])
                times = pd.concat([times, new_row], ignore_index=True)
    times['Dataset'] = pd.to_numeric(times['Dataset'], errors='coerce').astype('Int64')
    times = times.drop_duplicates()
    return times


def print_latex_table(df_pivot):
    formatted_df = df_pivot.applymap(lambda x: f"{x:.2f}" if pd.notnull(x) else "/")
    latex_lines = []
    latex_lines.append(r"\begin{table}[h!]")
    latex_lines.append(r"    \footnotesize")
    latex_lines.append(r"        \begin{tabular*}{\textwidth}{@{\extracolsep{3em}} c|ccc @{}}")
    latex_lines.append(r"        \toprule")
    latex_lines.append(r"        OpenML Dataset ID & " + " & ".join(df_pivot.columns) + r" \\")
    latex_lines.append(r"        \midrule")
    # Add table rows
    for dataset_id, row in formatted_df.iterrows():
        row_str = f"        {dataset_id} & " + " & ".join(row.values) + r" \\"
        latex_lines.append(row_str)
    # Finish LaTeX code
    latex_lines.append(r"        \bottomrule")
    latex_lines.append(r"    \end{tabular*}")
    latex_lines.append(r"    \caption{Overview of time needed in seconds of \fe{} methods on the \OpenML{} datasets}")
    latex_lines.append(r"    \label{tab:overview-time-openfe}")
    latex_lines.append(r"\end{table}")
    # Join all lines into one LaTeX string
    latex_code = "\n".join(latex_lines)
    # Print LaTeX table
    print(latex_code)


def plot_time(average_time_per_method, time_per_method, name):
    # Plot
    plt.figure(figsize=(12, 7))
    time_per_method.plot(kind='bar', color='skyblue', label='Total Time per FE Method')
    average_time_per_method.plot(kind='bar', width=0.3, color='orange', label='Average Time per FE Method')
    plt.legend()
    plt.xlabel("Method")
    plt.ylabel("Time in seconds")
    plt.yscale('log')
    plt.title("Time Usage of FE Methods")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("Time_FE_methods_" + name + ".png")
    plt.show()


def main():
    times = get_times()

    time_per_method = times.groupby("SM - Method")["Time"].sum().sort_values(ascending=False)
    average_recursion = time_per_method.values[0] / times[times["SM"] == "Recursion"]["Dataset"].nunique()
    average_best = time_per_method.values[1] / times[times["SM"] == "One-shot"]["Dataset"].nunique()
    average_time_per_method = pd.Series([average_recursion, average_best], index=["Recursion", "One-shot"])

    plot_time(average_time_per_method, time_per_method, "SM")

    times = add_openfe_data(times)
    times_filtered = times[times["Method"] != "d2v"]

    df_pivot = times_filtered.pivot(index="Dataset", columns="SM - Method", values="Time")
    df_pivot = df_pivot.sort_index()

    print_latex_table(df_pivot)

    time_per_method = times_filtered.groupby("SM")["Time"].sum().sort_values(ascending=False)
    time_per_method = time_per_method.reset_index("SM")
    time_per_method.drop(index=1, inplace=True)
    average_openfe = time_per_method["Time"].values[1] / times[times["SM"] == "OpenFE"]["Dataset"].nunique()
    average_time_per_method = pd.Series([average_recursion, average_openfe], index=["MetaFE", "OpenFE"])

    plot_time(average_time_per_method, time_per_method, "MetaFE_OpenFE")


if __name__ == "__main__":
    main()

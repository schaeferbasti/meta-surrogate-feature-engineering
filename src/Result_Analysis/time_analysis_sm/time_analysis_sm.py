import os
import re

import pandas as pd
from matplotlib import pyplot as plt

def main():
    times = pd.DataFrame(columns=["Method", "Dataset", "Task", "Time"])

    log_files = os.listdir()

    for log_file in log_files:
        if log_file.endswith('.out'):
            with open(log_file, "r") as f:
                lines = f.readlines()

            method = log_file.split("_")[0]
            if method == "Impr":
                method = "Recursion"
            elif method == "Best":
                method = "One-shot"
            dataset = log_file.split("_")[0].split("-")[-1]

            comp_sentence = "Time for creating Comparison Result Matrix: "
            timeout_sentence = "[Monitor] Time limit exceeded"
            meta_sentence = "Method:"

            comp_time = None
            timeout = False

            for i, line in enumerate(lines):
                if timeout_sentence in line:
                    timeout = True
                    break  # no need to continue reading if timed out
                if line.startswith(meta_sentence):
                    parts = line.strip().split(",")
                    for part in parts:
                        if "Dataset" in part:
                            dataset = part.split(":")[1].strip()

                if line.startswith(comp_sentence):
                    # Check if the next line exists and starts with expected string
                    if i + 1 < len(lines) and lines[i + 1].startswith("Allocated memory"):
                        try:
                            comp_time = float(line.split(comp_sentence)[1].strip())
                        except ValueError:
                            comp_time = None  # just in case parsing fails

            if timeout:
                # Timeout case
                new_row = pd.DataFrame([{
                    "Method": method,
                    "Dataset": dataset,
                    "Task": "Calculate Comparison Matrix",
                    "Time": 7200.0
                }])
                times = pd.concat([times, new_row], ignore_index=True)
            elif comp_time is not None:
                # Successful case with valid comp_time
                new_row = pd.DataFrame([{
                    "Method": method,
                    "Dataset": dataset,
                    "Task": "Calculate Comparison Matrix",
                    "Time": comp_time
                }])
                times = pd.concat([times, new_row], ignore_index=True)
    times = times.drop_duplicates()

    time_per_method = times.groupby("Method")["Time"].sum().sort_values(ascending=False)
    average_recursion = time_per_method.values[0] / times[times["Method"] == "Recursion"]["Dataset"].nunique()
    average_best = time_per_method.values[1] / times[times["Method"] == "One-shot"]["Dataset"].nunique()
    average_time_per_method = pd.Series([average_recursion, average_best], index=["Recursion", "Best"])


    # Plot
    plt.figure(figsize=(12, 7))
    time_per_method.plot(kind='bar', color='skyblue', label='Total Time for 12 Datasets per Surrogate Model')
    average_time_per_method.plot(kind='bar', width=0.3, color='orange', label='Average Time over 12 Datasets per Surrogate Model')
    plt.legend()
    plt.xlabel("Method")
    plt.ylabel("Time in seconds")
    plt.title("Time Usage of Surrogate Models")
    plt.xticks(rotation=90, ha="right")
    # plt.yscale('log')
    plt.tight_layout()
    plt.savefig("Time_SM_oneshot_recursive.png")
    plt.show()

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
                "Method": method,
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
                    "Method": method,
                    "Dataset": dataset,
                    "Task": "FE",
                    "Time": time
                }])
                times = pd.concat([times, new_row], ignore_index=True)

    times['Dataset'] = pd.to_numeric(times['Dataset'], errors='coerce').astype('Int64')
    times = times.drop_duplicates()

    df_pivot = times.pivot(index="Dataset", columns="Method", values="Time")
    df_pivot = df_pivot.sort_index()
    df_pivot = df_pivot[["Recursion", "One-shot", "OpenFE"]]

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
        row_str = f"        {dataset_id} & " + " & ".join(row.values) + r" \\ \midrule"
        latex_lines.append(row_str)

    # Finish LaTeX code
    latex_lines.append(r"    \end{tabular*}")
    latex_lines.append(r"    \caption{Overview of time needed in seconds of \fe{} methods on the \OpenML{} datasets}")
    latex_lines.append(r"    \label{tab:overview-time-openfe}")
    latex_lines.append(r"\end{table}")

    # Join all lines into one LaTeX string
    latex_code = "\n".join(latex_lines)

    # Print LaTeX table
    print(latex_code)

    time_per_method = times.groupby("Method")["Time"].sum().sort_values(ascending=False)
    print(time_per_method)
    average_best = time_per_method.values[0] / times[times["Method"] == "Recursion"]["Dataset"].nunique()
    average_recursion = time_per_method.values[2] / times[times["Method"] == "Recursion"]["Dataset"].nunique()
    average_openfe = time_per_method.values[1] / times[times["Method"] == "OpenFE"]["Dataset"].nunique()
    average_time_per_method = pd.Series([average_recursion, average_best, average_openfe], index=["Recursion", "One-shot", "OpenFE"])

    # Plot
    plt.figure(figsize=(12, 7))
    time_per_method.plot(kind='bar', color='skyblue', label='Total Time per FE Method')
    average_time_per_method.plot(kind='bar', width=0.3, color='orange', label='Average Time per FE Method')
    plt.legend()
    plt.xlabel("Method")
    plt.ylabel("Time in seconds")
    plt.title("Time Usage of FE Methods")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("Time_FE_methods.png")
    plt.show()


if __name__ == "__main__":
    main()

import os
import re

import openml
import pandas as pd
from matplotlib import pyplot as plt

from src.Apply_and_Test.analyse_results import insert_line_breaks


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
                    "Time": 3600.0
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

    time_per_method = times.groupby("Method")["Time"].sum().sort_values(ascending=False)
    average_recursion = time_per_method.values[0] / 12
    average_best = time_per_method.values[1] / 12
    average_time_per_method = pd.Series([average_recursion, average_best], index=["Recursion", "Best"])

    # Plot
    plt.figure(figsize=(10, 6))
    time_per_method.plot(kind='bar', color='skyblue', label='Total Time for 12 Datasets per Surrogate Model')
    average_time_per_method.plot(kind='bar', width=0.3, color='orange', label='Average Time over 12 Datasets per Surrogate Model')
    plt.legend()
    plt.xlabel("Method")
    plt.ylabel("Time in seconds")
    plt.title("Time Usage of Surrogate Models")
    plt.xticks(rotation=45, ha="right")
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

    times = times.drop_duplicates()
    time_per_method = times.groupby("Method")["Time"].sum().sort_values(ascending=False)
    average_openfe = time_per_method.values[0] / 12
    average_recursion = time_per_method.values[1] / 10
    average_best = time_per_method.values[2] / 7
    average_time_per_method = pd.Series([average_openfe, average_recursion, average_best], index=["OpenFE", "Recursion", "Best"])

    # Plot
    plt.figure(figsize=(10, 6))
    time_per_method.plot(kind='bar', color='skyblue', label='Total Time for 12 Datasets per FE Method')
    average_time_per_method.plot(kind='bar', width=0.3, color='orange', label='Average Time over 12 Datasets per FE Method')
    plt.legend()
    plt.xlabel("Method")
    plt.ylabel("Time in seconds")
    plt.title("Time Usage of FE Methods")
    plt.xticks(rotation=45, ha="right")
    # plt.yscale('log')
    plt.tight_layout()
    plt.savefig("Time_FE_methods.png")
    plt.show()


if __name__ == "__main__":
    main()

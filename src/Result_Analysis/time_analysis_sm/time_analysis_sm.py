import os

import openml
import pandas as pd
from matplotlib import pyplot as plt

from src.Apply_and_Test.analyse_results import insert_line_breaks


def main():
    times = pd.DataFrame(columns=["Method", "Dataset", "Task", "Time"])

    log_files = os.listdir()

    for log_file in log_files:
        if log_file.endswith('.out'):
            f = open(log_file, "r")
            lines = f.readlines()
            method = log_file.split("_")[0]
            if method == "Impr":
                method = "Recursion"
            elif method == "Best":
                method = "One-shot"
            dataset = log_file.split("_")[0].split("-")[-1]

            for line in lines:
                comp_sentence = "Time for creating Comparison Result Matrix: "
                pred_sentence = "Time for Predicting Improvement using CatBoost: "
                if line.startswith(comp_sentence):
                    time = line.split(comp_sentence)[1]
                    new_row = pd.DataFrame(columns=["Method", "Dataset", "Task", "Time"])
                    new_row.loc[len(new_row)] = [method, dataset, "Calculate Comparison Matrix", float(time)]
                    times = pd.concat([times, new_row], ignore_index=True)
                elif line.startswith(pred_sentence):
                    time = line.split(pred_sentence)[1]
                    new_row = pd.DataFrame(columns=["Method", "Dataset", "Task", "Time"])
                    new_row.loc[len(new_row)] = [method, dataset, "Predict Improvement", float(time)]
                    times = pd.concat([times, new_row], ignore_index=True)

    time_per_method = times.groupby("Method")["Time"].sum().sort_values(ascending=False)
    average_time_per_method = times.groupby("Method")["Time"].mean().sort_values(ascending=False)

    # Plot
    plt.figure(figsize=(10, 6))
    time_per_method.plot(kind='bar', color='skyblue', label='Total Time per Method')
    average_time_per_method.plot(kind='bar', width=0.3, color='orange', label='Average Time per Method')
    plt.legend()
    plt.xlabel("Method")
    plt.ylabel("Time in seconds")
    plt.title("Time per Method")
    plt.xticks(rotation=45, ha="right")
    # plt.yscale('log')
    plt.tight_layout()
    plt.savefig("Time_for_SM_per_Method.png")
    plt.show()


if __name__ == "__main__":
    main()

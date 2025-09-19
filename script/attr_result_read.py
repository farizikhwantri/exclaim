#!/usr/bin/env python
import sys
import re
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def parse_attr_file(file_path):
    results = {}
    current_model = None
    current_section = None
    current_explainer = None
    model_pattern = re.compile(r"^\S.+$")
    section_map = {
        "Model Attributions:": "overall",
        "Correct Attributions:": "correct",
        "Incorrect Attributions:": "incorrect"
    }
    explainer_pattern = re.compile(r"^Explainer:\s*(.+)$")
    metric_pattern = re.compile(r"^Metric:\s*([^,]+),\s*Mean Score:\s*([\-\d\.]+),\s*Std:\s*([\-\d\.]+)")
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if ":" not in line and not line.startswith("-"):
                if "Permutation Test Results" in line:
                    break
                current_model = line
                results[current_model] = {}
                current_section = None
                current_explainer = None
                continue
            for key, section in section_map.items():
                if line.startswith(key):
                    current_section = section
                    continue
            m = explainer_pattern.match(line)
            if m:
                current_explainer = m.group(1).strip()
                if current_model is not None:
                    if current_explainer not in results[current_model]:
                        results[current_model][current_explainer] = {"overall": {}, "correct": {}, "incorrect": {}}
                continue
            m = metric_pattern.match(line)
            if m and current_model and current_explainer and current_section:
                metric_name = m.group(1).strip()
                mean_score = float(m.group(2))
                std_score = float(m.group(3))
                results[current_model][current_explainer][current_section][metric_name] = {
                    "mean": mean_score,
                    "std": std_score
                }
                continue
            if line.startswith("-" * 5):
                continue
    return results

def results_to_table(results):
    rows = []
    for model, expl_dict in results.items():
        for explainer, sections in expl_dict.items():
            metrics = set()
            for sect in sections.values():
                metrics.update(sect.keys())
            for metric in metrics:
                overall = sections["overall"].get(metric, {"mean": np.nan, "std": np.nan})
                correct = sections["correct"].get(metric, {"mean": np.nan, "std": np.nan})
                incorrect = sections["incorrect"].get(metric, {"mean": np.nan, "std": np.nan})
                rows.append({
                    "Model": model,
                    "Explainer": explainer,
                    "Metric": metric,
                    "Overall_Mean": overall["mean"],
                    "Overall_Std": overall["std"],
                    "Correct_Mean": correct["mean"],
                    "Correct_Std": correct["std"],
                    "Incorrect_Mean": incorrect["mean"],
                    "Incorrect_Std": incorrect["std"]
                })
    df = pd.DataFrame(rows)
    return df

def write_grouped_table_to_csv(df, output_file):
    df["Overall"] = df.apply(lambda row: f"{row['Overall_Mean']:.3f}({row['Overall_Std']:.2f})", axis=1)
    df["Correct"] = df.apply(lambda row: f"{row['Correct_Mean']:.3f}({row['Correct_Std']:.2f})", axis=1)
    df["Incorrect"] = df.apply(lambda row: f"{row['Incorrect_Mean']:.3f}({row['Incorrect_Std']:.2f})", axis=1)
    out_cols = ["Model", "Explainer", "Metric", "Overall", "Correct", "Incorrect"]
    # make sure to apply the format to the mean and std columns
    # df["Overall_Mean"] = df["Overall_Mean"].apply(lambda x: f"{x:.3f}")
    # df["Overall_Std"] = df["Overall_Std"].apply(lambda x: f"{x:.2f}")
    # df["Correct_Mean"] = df["Correct_Mean"].apply(lambda x: f"{x:.3f}")
    # df["Correct_Std"] = df["Correct_Std"].apply(lambda x: f"{x:.2f}")
    # df["Incorrect_Mean"] = df["Incorrect_Mean"].apply(lambda x: f"{x:.3f}")
    # df["Incorrect_Std"] = df["Incorrect_Std"].apply(lambda x: f"{x:.2f}")
    # out_cols = ["Model", "Explainer", "Metric", "Overall_Mean", "Overall_Std", "Correct_Mean", "Correct_Std", "Incorrect_Mean", "Incorrect_Std"]
    with open(output_file, "w", newline="") as f:
        metrics = df["Metric"].unique()
        for metric in metrics:
            f.write(f"Metric: {metric}\n")
            group = df[df["Metric"] == metric][out_cols]
            group.to_csv(f, index=False)
            f.write("\n")

def main():
    file_path = sys.argv[1] if len(sys.argv) > 1 else "attr_result.txt"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    results = parse_attr_file(file_path)
    df = results_to_table(results)
    print(df)
    output_file = sys.argv[2] if len(sys.argv) > 2 else "attribution_summary.csv"
    write_grouped_table_to_csv(df, output_file)
    print(f"Grouped results saved to {output_file}")

if __name__ == "__main__":
    main()

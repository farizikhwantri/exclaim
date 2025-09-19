#!/usr/bin/env python3
import pandas as pd

# Read the CSV file while skipping comment lines (lines starting with "//")
# df = pd.read_csv("train_set.csv", comment="//")
# read from argument
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--input_csv", type=str, required=True)
args = parser.parse_args()

df = pd.read_csv(args.input_csv, delimiter=',')

# Compute basic dataset statistics
stats = {}
stats["Total Rows"] = len(df)
for col in df.columns:
    stats[f"Unique {col}"] = df[col].nunique()

# Prepare LaTeX table code (using booktabs style)
latex_lines = []
latex_lines.append(r"\begin{table}[ht]")
latex_lines.append(r"    \centering")
latex_lines.append(r"    \begin{tabular}{lr}")
latex_lines.append(r"        \toprule")
latex_lines.append(r"        \textbf{Metric} & \textbf{Value} \\")
latex_lines.append(r"        \midrule")
for metric, value in stats.items():
    latex_lines.append(f"        {metric} & {value} \\\\")
latex_lines.append(r"        \bottomrule")
latex_lines.append(r"    \end{tabular}")
latex_lines.append(r"    \caption{Dataset Statistics for train\_set.csv}")
latex_lines.append(r"    \label{tab:dataset_stats}")
latex_lines.append(r"\end{table}")

# Print the LaTeX table code
print("\n".join(latex_lines))
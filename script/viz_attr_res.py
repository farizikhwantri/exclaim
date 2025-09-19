import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re

# Read the TSV file (example: aopc_compr.tsv)
input_file = sys.argv[1] if len(sys.argv) > 1 else "aopc_compr.tsv"
metric_name = sys.argv[2] if len(sys.argv) > 2 else "AOPC Comprehensiveness"
output_dir = sys.argv[3] if len(sys.argv) > 3 else "data/dpa-multi"
df = pd.read_csv(input_file, sep="\t", encoding="utf-8")

# Function to extract mean and std from a string like "0.276(0.26)"
def extract_mean_std(s):
    m = re.search(r'([\-\d\.]+)\(([\-\d\.]+)\)', s)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None, None

# Extract Overall mean and std into new numeric columns
df[['Overall_Mean','Overall_Std']] = df['Overall'].apply(lambda s: pd.Series(extract_mean_std(s)))

# Group by Model so bars for the same model are together.
models = df["Model"].unique()
explainers = df["Explainer"].unique()

# Create a color mapping for explainers using a Seaborn palette.
palette = sns.color_palette("Set2", len(explainers))
color_dict = dict(zip(explainers, palette))

# Define bar width per explainer within a model group.
n_explainers = len(explainers)
bar_width = 0.8 / n_explainers
x_indices = np.arange(len(models))  # one tick per model

plt.figure(figsize=(12, 6))
plt.rcParams.update({'font.size': 14})

# Loop over each explainer and plot the bars for all models.
for i, explainer in enumerate(explainers):
    vals = []
    errs = []
    # For each model, extract the row with this explainer, if present.
    for model in models:
        row = df[(df["Model"] == model) & (df["Explainer"] == explainer)]
        if not row.empty:
            vals.append(row["Overall_Mean"].values[0])
            errs.append(row["Overall_Std"].values[0])
        else:
            vals.append(np.nan)
            errs.append(np.nan)
            
    # Compute the x positions: offset each explainer bar within the model group.
    offsets = x_indices - 0.4 + i * bar_width + bar_width/2
    plt.bar(offsets, vals, width=bar_width, yerr=errs, capsize=5, 
            color=color_dict[explainer], label=explainer)

plt.xticks(x_indices, models, rotation=45, ha='right')
# plt.ylabel("Overall AOPC ")
# plt.title("Overall Attribution Scores by Model and Explainer (Mean ± Std)")
# plt.legend(title="Explainer")
plt.ylabel(f"{metric_name}")
plt.title(f"{metric_name} by Model and Explainer (Mean ± Std)")
plt.legend(title="Post-hoc Explainer", bbox_to_anchor=(1.05, 1), loc='upper left')
# make the legend outside the in the bottom center
# plt.legend(title="Post-hoc Explainer", bbox_to_anchor=(0.5, -0.15), loc='lower center', ncol=4)
plt.tight_layout()
# plt.show()
# save the figure in pdf format
# output_dir = "data/dpa-multi"
input_name = input_file.split("/")[-1].split(".")[0]
# make the text in the figure larger
# plt.show()
plt.savefig(f"{output_dir}/{input_name}_mean-overall.pdf", dpi=300)
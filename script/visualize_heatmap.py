# import ast
# import re
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Path to the merged output file
# filepath = '/Users/fariz/repositories/REng-xai-cert/data/sac_gdpr/merged_gdpr.out'

# # Read the whole file
# with open(filepath, 'r', encoding='utf-8') as f:
#     content = f.read()

# # --- 1. Parse intra-model dictionary ---
# # Look for the line "average graph edit distance intra-model per model" followed by a dictionary.
# intra_dict = {}
# intra_pattern = r"average graph edit distance intra-model per model\s*(\{.*?\})"
# intra_match = re.search(intra_pattern, content, re.DOTALL)
# if intra_match:
#     intra_str = intra_match.group(1)
#     try:
#         # Use ast.literal_eval to convert the string to a dictionary.
#         intra_data = ast.literal_eval(intra_str)
#         # Compute average per model (if list non-empty)
#         for model, vals in intra_data.items():
#             if vals:
#                 intra_dict[model] = np.mean(vals)
#             else:
#                 intra_dict[model] = np.nan
#     except Exception as e:
#         print("Error parsing intra-model data:", e)
# else:
#     print("Intra-model dictionary not found.")

# # --- 2. Parse inter-model lines ---
# # They have the pattern: ( 'modelA', 'modelB' )  number  value
# inter_dict = {}
# inter_pattern = r"\('([^']+)',\s*'([^']+)'\)\s+\d+\s+([\d\.]+)"
# for match in re.finditer(inter_pattern, content):
#     model_a, model_b, value = match.groups()
#     value = float(value)
#     # Create a frozenset as key so that order does not matter.
#     key = frozenset({model_a, model_b})
#     inter_dict[key] = value

# # --- 3. Build the full sorted list of models ---
# # We'll use the models from intra_dict; you might also add models found in inter_dict.
# models = sorted(intra_dict.keys())

# n = len(models)
# matrix = np.zeros((n, n))

# # Fill diagonal from intra-model averages
# for i, mod in enumerate(models):
#     matrix[i, i] = intra_dict.get(mod, np.nan)

# # Fill off-diagonals using the inter_dict lookup.
# for i in range(n):
#     for j in range(n):
#         if i == j:
#             continue
#         key = frozenset({models[i], models[j]})
#         # If the pair exists, set the value; otherwise, use np.nan.
#         matrix[i, j] = inter_dict.get(key, np.nan)

# # --- 4. Generate the heatmap ---
# plt.figure(figsize=(10, 8))
# ax = sns.heatmap(matrix, annot=True, fmt=".2f", xticklabels=models, yticklabels=models,
#                  cmap="viridis", mask=np.isnan(matrix))
# plt.title("Graph Edit Distance Heatmap\n(Diagonal = Intra-model; Off-diagonals = Inter-model)")
# plt.xticks(rotation=45, ha="right")
# plt.yticks(rotation=0)
# plt.tight_layout()
# plt.show()

import ast
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Parse the merged output file ---
# filepath = '/Users/fariz/repositories/REng-xai-cert/data/sac_gdpr/merged_gdpr.out'
# read filepath from command line argument or set it directly
import sys
filepath = sys.argv[1] if len(sys.argv) > 1 else '/Users/fariz/repositories/REng-xai-cert/data/sac_gdpr/merged_gdpr.out'

with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# Parse intra-model data
intra_dict = {}
intra_pattern = r"average graph edit distance intra-model per model\s*(\{.*?\})"
intra_match = re.search(intra_pattern, content, re.DOTALL)
if intra_match:
    intra_str = intra_match.group(1)
    try:
        intra_data = ast.literal_eval(intra_str)
        for model, vals in intra_data.items():
            print(model, len(vals))
            intra_dict[model] = np.mean(vals) if vals else np.nan
    except Exception as e:
        print("Error parsing intra-model data:", e)
else:
    print("Intra-model dictionary not found.")

# Parse inter-model data
inter_dict = {}
inter_pattern = r"\('([^']+)',\s*'([^']+)'\)\s+\d+\s+([\d\.]+)"
for match in re.finditer(inter_pattern, content):
    # ignore model gpt-4-8k
    model_a, model_b, value = match.groups()
    value = float(value)
    key = frozenset({model_a, model_b})
    inter_dict[key] = value

# --- 2. Define a mapping from model names to their parameter size groups ---
# Adjust these values as needed.
model_params = {
    # 'gpt-4o': 'UNK',
    'Qwen2.5-0.5B-Instruct': '0.5B',
    'gemma-2-9b-it': '9B',
    'phi-4': '14B',
    'Phi-3-medium-4k-instruct': '14B',
    'Phi-3-medium-128k-instruct': '14B',
    'Llama-3.2-3B-Instruct': '3B',
    'Qwen2.5-14B-Instruct': '14B',
    'Qwen2.5-14B-Instruct-1M': '14B',
    'gpt-35-turbo-4k': '200B',
    'gemma-7b-it': '7B',
    'Qwen2.5-7B-Instruct': '7B',
    'Qwen2.5-7B': '7B',
    'gpt-4o-8k': '400B',  
    # 'gpt-4o': '400B', 
    'Llama-3.1-70B-Instruct': '70B',
    'Qwen2.5-72B-Instruct': '72B',
    'Llama-3.1-8B-Instruct': '8B',
    'gemma-2-27b-it': '27B'
}

# Helper function to convert parameter strings to a numeric value.
def param_value(model):
    if model in model_params:
        try:
            # Remove any non-digit characters (like 'B') and convert.
            return float(''.join(ch for ch in model_params[model] if ch.isdigit()))
        except Exception as e:
            return float('inf')
    else:
        return float('inf')

# --- 3. Build the sorted list of models based on parameter size: smallest first; unknown at the end.
models = sorted(intra_dict.keys(), key=lambda m: param_value(m))
# remove the instruct suffix from models

n = len(models)
matrix = np.zeros((n, n))

# Fill diagonal from intra-model averages.
for i, mod in enumerate(models):
    matrix[i, i] = intra_dict.get(mod, np.nan)

# Fill off-diagonals using the inter_dict lookup.
for i in range(n):
    for j in range(n):
        if i == j:
            continue
        key = frozenset({models[i], models[j]})
        matrix[i, j] = inter_dict.get(key, np.nan)


models = [re.sub(r'-Instruct$', '', m) for m in models]

# --- 4. Generate the heatmap ---
plt.figure(figsize=(10, 8))
ax = sns.heatmap(matrix, annot=True, fmt=".2f", xticklabels=models, yticklabels=models,
                 cmap="viridis", mask=np.isnan(matrix))
# plt.title("Graph Edit Distance Heatmap\n(Diagonal = Intra-model; Off-diagonals = Inter-model)")
# plt.xticks(rotation=45, ha="right")
# plt.yticks(rotation=0)
plt.title("Graph Edit Distance Heatmap\n(Diagonal = Intra-model; Off-diagonals = Inter-model)", fontsize=22)
plt.xticks(rotation=45, ha="right", fontsize=16)
plt.yticks(rotation=0, fontsize=16)
plt.tight_layout()
plt.show()

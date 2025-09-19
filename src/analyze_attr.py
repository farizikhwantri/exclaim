#!/usr/bin/env python
import os
import pickle
import argparse

import tqdm

import random

import numpy as np
import torch
from transformers import AutoTokenizer
from peft import TaskType

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from ferret.evaluators import Explanation
from ferret.evaluators.faithfulness_measures import Evaluation

# from sklearn.metrics import accuracy_score

from pipeline import construct_model
from eval_split import load_lora_model

# --- Helper Functions ---

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# check if cuda or mps is available
if torch.cuda.is_available():
    DEVICE = 'cuda'
elif torch.backends.mps.is_available():
    DEVICE = 'mps'
else:
    DEVICE = 'cpu'

explainer_names_map = {
    "Integrated Gradient (x Input)": "IG",
    "Gradient (x Input)": "GradXInput",
    "LIME": "LIME",
    "Partition SHAP": "PartSHAP",
}

def map_explainer_name(explainer_name):
    """Map the explainer name to a shorter version."""
    if explainer_name in explainer_names_map:
        return explainer_names_map[explainer_name]
    return explainer_name

print(f"Using device: {DEVICE}")

def read_pickle(file_path):
    """Load a pickle file."""
    with open(file_path, "rb") as f:
        return pickle.load(f)

def collect_results(results, res_explainer):
    # print(len(results))
    for result in results:
        for explainer_eval in result:
            explanation = explainer_eval.explanation
            # if random.random() < 0.1:
            #     pass
            assert isinstance(explanation, Explanation)
            if explanation.explainer not in res_explainer:
                # skip tau_corr_loo
                res_explainer[explanation.explainer] = {}

            evaluation = explainer_eval.evaluation_scores
            assert isinstance(evaluation[0], Evaluation)
            for eval in evaluation:
                # skip tau_corr_loo
                if eval.name not in res_explainer[explanation.explainer]:
                    res_explainer[explanation.explainer][eval.name] = []
                res_explainer[explanation.explainer][eval.name].append(eval.score)

    return res_explainer

def print_attributions(attributions):
    # Print the collected results.
    for explainer, metrics in attributions.items():
        print(f"Explainer: {explainer}")
        for metric, scores in metrics.items():
            # print(f"  Metric: {metric}, Scores: {scores}")
            # print(f"  Metric: {metric}, Mean Score: {np.mean(scores)}, Std: {np.std(scores)}")
            print(f"  Metric: {metric}, Mean Score: {np.mean(scores):.4f}, Std: {np.std(scores):.4f}")
        print("\n------------------------\n")

def permutation_test(x, y, num_permutations=10000, alternative='greater'):
    """
    Perform a permutation test comparing two samples x and y.
    
    alternative: 'greater' means H1: mean(x) > mean(y);
                 'less'    means H1: mean(x) < mean(y);
                 'two-sided' means H1: mean(x) != mean(y)
    
    Returns the observed difference in means (mean(x) - mean(y)) and a p-value.
    """
    x = np.array(x)
    y = np.array(y)
    obs_diff = np.mean(x) - np.mean(y)
    combined = np.concatenate([x, y])
    count = 0
    n_x = len(x)
    for _ in range(num_permutations):
        np.random.shuffle(combined)
        new_x = combined[:n_x]
        new_y = combined[n_x:]
        perm_diff = np.mean(new_x) - np.mean(new_y)
        if alternative == 'greater':
            if perm_diff >= obs_diff:
                count += 1
        elif alternative == 'less':
            if perm_diff <= obs_diff:
                count += 1
        else:  # two-sided
            if abs(perm_diff) >= abs(obs_diff):
                count += 1
    p_value = count / num_permutations
    return obs_diff, p_value

def visualize_attr_boxplot(cor_attributions, inc_attributions):
    """
    Create boxplots comparing attribution scores for correct and incorrect predictions.
    This function expects:
      - cor_attributions: dict { explainer: { metric: [score1, score2, ...], ... }, ... }
      - inc_attributions: dict with the same structure as above.
      
    The 'taucorr_loo' metric is ignored.
    """

    # Build a DataFrame for plotting.
    rows = []
    # Process correct attributions.
    for explainer, metrics in cor_attributions.items():
        explainer = map_explainer_name(explainer)
        for metric, scores in metrics.items():
            if metric == "taucorr_loo":
                continue
            for s in scores:
                rows.append({
                    "Explainer": explainer,
                    "Metric": metric,
                    "Score": s,
                    "Prediction": "Correct"
                })
    # Process incorrect attributions.
    for explainer, metrics in inc_attributions.items():
        explainer = map_explainer_name(explainer)
        for metric, scores in metrics.items():
            if metric == "taucorr_loo":
                continue
            for s in scores:
                rows.append({
                    "Explainer": explainer,
                    "Metric": metric,
                    "Score": s,
                    "Prediction": "Incorrect"
                })
                
    df = pd.DataFrame(rows)
    
    # If there are multiple metrics, create a separate subplot for each.
    unique_metrics = df["Metric"].unique()
    num_metrics = len(unique_metrics)
    
    fig, axs = plt.subplots(1, num_metrics, figsize=(6 * num_metrics, 6), squeeze=False)
    
    for idx, metric in enumerate(unique_metrics):
        ax = axs[0, idx]
        # Subset the DataFrame for this metric.
        df_metric = df[df["Metric"] == metric]
        sns.boxplot(x="Explainer", y="Score", hue="Prediction", data=df_metric, ax=ax, palette="Set2")
        ax.set_title(f"{metric} of Correct vs Incorrect Predictions")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_xlabel("Explainer")
        ax.set_ylabel(f"{metric} Score")
    
    plt.tight_layout()
    plt.show()
    
# Example usage; at the end of your main script after collecting cor_attributions and inc_attributions:
# visualize_attr_boxplot(cor_attributions, inc_attributions)

# --- Main Script ---

def main(args):
    # Load pretrained model and tokenizer.
    # model_name = "distilbert-base-uncased-finetuned-sst-2-english"  # Example: adjust as needed.
    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    model = construct_model(
        model_name=model_name,
        data_name='custom_nli',
        num_labels=2,).to(DEVICE)

    if args.load_lora:
        model = load_lora_model(model=model, 
                                target_modules=args.target_modules, 
                                task_type=TaskType.SEQ_CLS)
        
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, 'custom_nli')
    state_dict = torch.load(os.path.join(args.checkpoint_dir, "model.pth"), map_location=DEVICE)
    model.load_state_dict(state_dict=state_dict)
    model.eval()

    # Load attribution data from a pickle file.
    # Each entry in the file should contain:
    #   "example": { "text": <input text>, "label": <true label> }
    #   "correct_results": [ ... ] (list of evaluation objects if prediction is correct)
    #   "incorrect_results": [ ... ] (list of evaluation objects if prediction is incorrect)
    # pickle_file = "path/to/attributions.pkl"  # Replace with your actual file path.

    pickle_file = args.pickle_file
    if not os.path.exists(pickle_file):
        raise FileNotFoundError(f"Pickle file not found: {pickle_file}")
    data = read_pickle(pickle_file)

    print(f"Loaded {len(data)} instances from {pickle_file}")

    # pick 1000 instances from pickle file
    data = random.sample(data, 1000)

    predictions = []
    pred_scores = []
    texts = []
    true_labels = []    

    model_attributions = {}
    cor_attributions = {}
    inc_attributions = {}
    
    # Run model prediction on each instance coming from the attribution file.
    for instance in tqdm.tqdm(data):
        # print(f"Processing instance: {type(instance)}, {instance.keys()}")
        # Extract text and label from the instance.
        text = instance.get("text", "")
        label = instance.get("label")
        # print(f"Text: {text}, Label: {label}")
        if label is None:
            continue  # Skip if no ground truth.
        
        texts.append(text)
        true_labels.append(label)
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()
        pred_score = torch.softmax(outputs.logits, dim=1).max().item()
        pred_scores.append(pred_score)

        predictions.append(pred)
        
        # Attach prediction correctness to the instance data.
        # instance["correct"] = (pred == label)
        if pred == label and instance.get("correct_results"):
            # Collect correct results.
            model_attributions = collect_results(instance["correct_results"], model_attributions)
            cor_attributions = collect_results(instance["correct_results"], cor_attributions)
        elif pred != label and instance.get("incorrect_results"):
            # Collect incorrect results.
            model_attributions = collect_results(instance["incorrect_results"], model_attributions)
            inc_attributions = collect_results(instance["incorrect_results"], inc_attributions)

    # print the model predictions average attributions
    print("Model Attributions:")
    print_attributions(model_attributions)
    print("Correct Attributions:")
    print_attributions(cor_attributions)
    print("Incorrect Attributions:")
    print_attributions(inc_attributions)

    # After you have collected model_attributions, cor_attributions, and inc_attributions
    # you can conduct the permutation test for each explainer and metric.

    print("\nPermutation Test Results (H1: Correct > Incorrect):")
    for explainer in cor_attributions:
        # Check if there is a corresponding key in incorrect attributions.
        if explainer not in inc_attributions:
            print(f"Explainer: {explainer} - No incorrect results collected.")
            continue
        for metric in cor_attributions[explainer]:
            if metric not in inc_attributions[explainer]:
                print(f"Explainer: {explainer}, Metric: {metric} - Skipping, no incorrect scores available.")
                continue
            correct_scores = np.array(cor_attributions[explainer][metric])
            incorrect_scores = np.array(inc_attributions[explainer][metric])
            
            # Run the permutation test.
            alternative = 'greater'  # H1: mean(correct_scores) > mean(incorrect_scores)
            # if metric is sufficiency, then the alternative is less
            if metric == "aopc_suff":
                alternative = 'two-sided'        
            
            diff, p_val = permutation_test(correct_scores, incorrect_scores, num_permutations=10000, alternative=alternative)
            print(f"Explainer: {explainer}, Metric: {metric}")
            print(f"  Observed Difference (Correct - Incorrect): {diff:.4f}")
            print(f"  p-value: {p_val:.4f}\n")

    visualize_attr_boxplot(cor_attributions, inc_attributions)


    # for each attribution method, and its metrics conduct permutation test whether the attributions of correct > incorrect

    # for each attribution method, and its metrics get the score for the pred
    # make a sorted arg version of the results of the model attributions faithfulness metrics to access the results
    # for each attribution method, and its metrics get the score for the pred
    paired_faith_pred_scores = {}
    for explainer, metrics in model_attributions.items():
        print(f"Explainer: {explainer}")
        paired_faith_pred_scores[explainer] = {}
        for metric, scores in metrics.items():
            # print(f"  Metric: {metric}, Scores: {scores}")
            # print(f"  Metric: {metric}, Mean Score: {np.mean(scores)}, Std: {np.std(scores)}")
            assert len(scores) == len(predictions)
            paired_faith_pred_scores[explainer][metric] = []
            for i in range(len(texts)):
                # calculate the pred =? true_label
                pred = predictions[i]
                true_label = true_labels[i]
                pred_score = pred_scores[i]
                text = texts[i]
                # one if true_label == pred else zero
                res = 1 if true_label == pred else 0
                paired_faith_pred_scores[explainer][metric].append((res, pred_score, scores[i]))
       

if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Analyze attribution results from a pickle file.")
    args.add_argument(
        "--pickle_file", type=str, required=True, help="Path to the pickle file containing attribution results."
    )
    args.add_argument("--model_name", type=str, default="distilbert-base-uncased-finetuned-sst-2-english", 
                      help="Pretrained model name.")
    args.add_argument("--checkpoint_dir", type=str, default=None, help="Path to the checkpoint directory.")
    args.add_argument("--load_lora", action="store_true", help="Load LoRA model")
    args.set_defaults(load_lora=False)
    args.add_argument("--target_modules", type=str, nargs="+", default=None, help="Target modules to apply LoRA")
    args.add_argument("--task_type", type=str, default="SEQ_CLS", help="Task type for LoRA model.")
    args.add_argument("--dataset_name", type=str, default="custom_nli", help="Dataset name.")
    args = args.parse_args()
    main(args)
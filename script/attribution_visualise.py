import argparse
import pickle

import re

import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
# Import classes from ferret (ensure your PYTHONPATH is set appropriately)
from ferret.evaluators.evaluation import ExplanationEvaluation, Explanation, Evaluation


def read_pickle(file_path):
    return pickle.load(open(file_path, "rb"))


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze the pickle file')
    # Allow specifying multiple files; we use the first one for demonstration.
    parser.add_argument('--file_path', nargs='+', type=str, required=True, help='Path to the pickle file')
    return parser.parse_args()


def collect_results(results, res_explainer):
    """
    For every evaluation instance in the results, we collect a tuple of:
       (evaluation_score, explanation.scores, explanation.text)
    keyed by explainer and evaluation metric name.
    """
    for result in results:
        print("Found", len(result), "evaluations in one result block")
        for explainer_eval in result:
            explanation = explainer_eval.explanation
            # Ensure the explanation is an instance of the Explanation class.
            assert isinstance(explanation, Explanation)
            explainer = explanation.explainer
            if explainer not in res_explainer:
                res_explainer[explainer] = {}
            evaluation = explainer_eval.evaluation_scores
            # Ensure there is at least one evaluation.
            assert len(evaluation) > 0 and hasattr(evaluation[0], "name")
            for eval_obj in evaluation:
                if eval_obj.name not in res_explainer[explainer]:
                    res_explainer[explainer][eval_obj.name] = []
                # Append a tuple (evaluation score, explanation scores, explanation text)
                res_explainer[explainer][eval_obj.name].append(
                    (eval_obj.score, explanation.scores, explanation.text)
                    # (eval_obj.score, explanation.scores, explanation.tokens)
                )
    return res_explainer

def visualize_explanation_heatmap(importance_scores, explanation_text):
    """
    Visualizes the post-hoc explanation importance scores as two heatmaps.
    The explanation text is split into sentences using punctuation.
    All sentences except the last are concatenated as the premise; the last sentence is the hypothesis.
    Importance scores are allocated to tokens sequentially.
    Both heatmaps apply min–max normalization separately and do not display cell annotations.
    """
    # Split the explanation text into sentences using punctuation.
    sentences = re.split(r'(?<=[.!?])\s+', explanation_text.strip())
    
    # If there is only one sentence, fallback to a single heatmap.
    if len(sentences) < 2:
        print("Not enough sentences to separate premise and hypothesis; displaying a single heatmap.")
        _visualize_single_heatmap(importance_scores, explanation_text)
        return

    # Define premise (all but last) and hypothesis (last sentence)
    premise_text = " ".join(sentences[:-1]).strip()
    hypothesis_text = sentences[-1].strip()

    # Tokenize each part.
    tokens_premise = premise_text.split()
    tokens_hypothesis = hypothesis_text.split()

    # Compute the number of tokens required
    total_tokens_needed = len(tokens_premise) + len(tokens_hypothesis)

    # If the number of importance scores is different than the needed count, adjust by taking the minimum length.
    if len(importance_scores) != total_tokens_needed:
        n = min(len(importance_scores), total_tokens_needed)
        # Re-split the tokens (cutting off extra tokens if needed)
        combined_tokens = (tokens_premise + tokens_hypothesis)[:n]
        # Recompute how many belong to premise based on original ratio.
        n_premise = min(len(tokens_premise), n)
        tokens_premise = combined_tokens[:n_premise]
        tokens_hypothesis = combined_tokens[n_premise:]
        importance_scores = importance_scores[:n]

    else:
        n_premise = len(tokens_premise)

    # Allocate importance scores.
    importance_premise = np.array(importance_scores[:n_premise], dtype=float)
    importance_hypothesis = np.array(importance_scores[n_premise:total_tokens_needed], dtype=float)

    # Apply min-max normalization to each separately.
    def normalize(scores):
        if scores.size == 0:
            return scores
        smin = scores.min()
        smax = scores.max()
        if smax - smin > 0:
            return (scores - smin) / (smax - smin)
        else:
            return scores
    norm_premise = normalize(importance_premise)
    norm_hypothesis = normalize(importance_hypothesis)

    # Prepare heatmap data.
    heatmap_data_premise = np.expand_dims(norm_premise, axis=0)
    heatmap_data_hypothesis = np.expand_dims(norm_hypothesis, axis=0)

    # Create subplots for premise and hypothesis.
    fig, axes = plt.subplots(2, 1, figsize=(max(len(tokens_premise), len(tokens_hypothesis))*0.5, 4))

    # Premise heatmap.
    sns.heatmap(heatmap_data_premise, annot=False, cmap="Blues",
                xticklabels=tokens_premise, yticklabels=["Premise"],
                cbar_kws={"label": "Normalized Score"}, ax=axes[0])
    axes[0].set_title("Premise")
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha="right")

    # Hypothesis heatmap.
    sns.heatmap(heatmap_data_hypothesis, annot=False, cmap="Blues",
                xticklabels=tokens_hypothesis, yticklabels=["Hypothesis"],
                cbar_kws={"label": "Normalized Score"}, ax=axes[1])
    axes[1].set_title("Hypothesis")
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    plt.show()

def _visualize_single_heatmap(importance_scores, explanation_text):
    """
    Fallback visualization: single-row heatmap.
    """
    tokens = explanation_text.split()
    if len(importance_scores) != len(tokens):
        n = min(len(importance_scores), len(tokens))
        tokens = tokens[:n]
        importance_scores = importance_scores[:n]
    scores = np.array(importance_scores, dtype=float)
    if scores.size == 0:
        print("Empty importance scores -- skipping visualization.")
        return
    smin = scores.min()
    smax = scores.max()
    normalized_scores = (scores - smin) / (smax - smin) if smax - smin > 0 else scores
    heatmap_data = np.expand_dims(normalized_scores, axis=0)
    plt.figure(figsize=(len(tokens)*0.5, 2))
    ax = sns.heatmap(heatmap_data, annot=False, cmap="coolwarm",
                     xticklabels=tokens, yticklabels=["Score"],
                     cbar_kws={"label": "Normalized Importance Score"})
    ax.set_title("Post-hoc Explanation Heatmap")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

def visualize_explanation_barplot(importance_scores, explanation_text):
    """
    Visualizes the post-hoc explanation importance scores as two bar plots.
    The explanation text is split into sentences using punctuation.
    All sentences except the last are concatenated as the premise; the last sentence is the hypothesis.
    Importance scores are allocated to tokens sequentially.
    Each part is normalized separately using min–max normalization.
    """
    # Split the explanation text into sentences.
    sentences = re.split(r'(?<=[.!?])\s+', explanation_text.strip())
    
    # If there is only one sentence, fallback to a single bar plot.
    if len(sentences) < 2:
        print("Not enough sentences to separate premise and hypothesis; displaying a single bar plot.")
        _visualize_single_barplot(importance_scores, explanation_text)
        return

    # Define premise (all but last) and hypothesis (last sentence)
    premise_text = " ".join(sentences[:-1]).strip()
    hypothesis_text = sentences[-1].strip()

    # Tokenize each part.
    tokens_premise = premise_text.split()
    tokens_hypothesis = hypothesis_text.split()

    # Compute the number of tokens required.
    total_tokens_needed = len(tokens_premise) + len(tokens_hypothesis)

    # If the number of importance scores differs from the required count, adjust by taking the minimum length.
    if len(importance_scores) != total_tokens_needed:
        n = min(len(importance_scores), total_tokens_needed)
        combined_tokens = (tokens_premise + tokens_hypothesis)[:n]
        n_premise = min(len(tokens_premise), n)
        tokens_premise = combined_tokens[:n_premise]
        tokens_hypothesis = combined_tokens[n_premise:]
        importance_scores = importance_scores[:n]
    else:
        n_premise = len(tokens_premise)

    # Allocate importance scores.
    importance_premise = np.array(importance_scores[:n_premise], dtype=float)
    importance_hypothesis = np.array(importance_scores[n_premise:total_tokens_needed], dtype=float)

    # Function for min–max normalization.
    def normalize(scores):
        if scores.size == 0:
            return scores
        smin = scores.min()
        smax = scores.max()
        if smax - smin > 0:
            return (scores - smin) / (smax - smin)
        else:
            return scores

    norm_premise = normalize(importance_premise)
    norm_hypothesis = normalize(importance_hypothesis)

    # Create subplots for premise and hypothesis.
    fig, axes = plt.subplots(2, 1, figsize=(max(len(tokens_premise), len(tokens_hypothesis)) * 0.5, 4))

    # Premise bar plot.
    ax = axes[0]
    ax.bar(range(len(tokens_premise)), norm_premise, color="blue")
    ax.set_xticks(range(len(tokens_premise)))
    ax.set_xticklabels(tokens_premise, rotation=45, ha="right")
    ax.set_ylabel("Normalized Score")
    ax.set_title("Premise")

    # Hypothesis bar plot.
    ax2 = axes[1]
    ax2.bar(range(len(tokens_hypothesis)), norm_hypothesis, color="blue")
    ax2.set_xticks(range(len(tokens_hypothesis)))
    ax2.set_xticklabels(tokens_hypothesis, rotation=45, ha="right")
    ax2.set_ylabel("Normalized Score")
    ax2.set_title("Hypothesis")

    plt.tight_layout()
    plt.show()

def _visualize_single_barplot(importance_scores, explanation_text):
    """
    Fallback visualization: single bar plot.
    """
    tokens = explanation_text.split()
    if len(importance_scores) != len(tokens):
        n = min(len(importance_scores), len(tokens))
        tokens = tokens[:n]
        importance_scores = importance_scores[:n]
    scores = np.array(importance_scores, dtype=float)
    if scores.size == 0:
        print("Empty importance scores -- skipping visualization.")
        return
    smin = scores.min()
    smax = scores.max()
    normalized_scores = (scores - smin) / (smax - smin) if smax - smin > 0 else scores
    plt.figure(figsize=(len(tokens) * 0.5, 2))
    plt.bar(range(len(tokens)), normalized_scores, color="blue")
    plt.xticks(range(len(tokens)), tokens, rotation=45, ha="right")
    plt.ylabel("Normalized Score")
    plt.title("Post-hoc Explanation Bar Plot")
    plt.tight_layout()
    plt.show()

def analyze_file(file_path):
    # load the pickle file
    data = read_pickle(file_path)
    
    cor_res_explainer = {}
    inc_res_explainer = {}
    
    print("Number of result blocks:", len(data))
    
    # Collect results from correct and incorrect blocks separately
    for res in data:
        correct_results = res.get('correct_results', [])
        cor_res_explainer = collect_results(correct_results, cor_res_explainer)
        incorrect_results = res.get('incorrect_results', [])
        inc_res_explainer = collect_results(incorrect_results, inc_res_explainer)
    
    # For demonstration, pick the first explainer and the first evaluation metric from correct results.
    if cor_res_explainer:
        first_explainer = list(cor_res_explainer.keys())[0]
        metric_dict = cor_res_explainer[first_explainer]
        if metric_dict:
            first_metric = list(metric_dict.keys())[0]
            eval_list = metric_dict[first_metric]
            if eval_list:
                # Sort by evaluation score (highest first)
                eval_list_sorted = sorted(eval_list, key=lambda x: x[0], reverse=True)
                top_instance = eval_list_sorted[0]
                score, importance_scores, explanation_text = top_instance
                print(f"Visualizing top instance for explainer '{first_explainer}' and metric '{first_metric}':")
                print(f"Evaluation score: {score}")
                print(f"Explanation text: {explanation_text}")
                visualize_explanation_heatmap(importance_scores, explanation_text)
                # visualize_explanation_barplot(importance_scores, explanation_text)
            else:
                print("No evaluation instances found for the metric.")
        else:
            print("No metric data found for the explainer.")
    else:
        print("No correct results available.")


if __name__ == '__main__':
    args = parse_args()
    # For demonstration, use the first file path provided.
    file_path = args.file_path[0]
    model_name = file_path.split('/')[-2]  # Just as an example extraction
    print("Model Name (extracted):", model_name)
    analyze_file(file_path)
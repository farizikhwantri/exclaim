import argparse
import pickle
import random
import copy

import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
# import ferret
from ferret.evaluators import Explanation, ExplanationWithRationale
from ferret.evaluators.evaluation import Evaluation

# read from pickle file
def read_pickle(file_path):
    return pickle.load(open(file_path, "rb"))


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze the pickle file')
    # parser.add_argument('--file_path', type=str, required=True, help='Path to the pickle file')
    # file path can be multiple files
    parser.add_argument('--file_path', nargs='+', type=str, required=True, help='Path to the pickle file')
    return parser.parse_args()

def visualize_boxplot(res_explainer, title, save_path=None):
    """
    Given a dictionary `res_explainer` in which:
      {
         'Explainer1': {
              'Metric1': [score, score, ...],
              'Metric2': [score, ...],
              ...
         },
         'Explainer2': { ... },
         ...
      }
    This function builds a DataFrame and creates a boxplot.
    """
    rows = []
    for explainer, evals in res_explainer.items():
        for eval_name, scores in evals.items():
            # Here we assume that scores is still a list of numeric scores.
            for s in scores:
                if eval_name != 'tau_corr_loo':
                    rows.append({
                        "Explainer": explainer,
                        "Metric": eval_name,
                        "Score": s
                    })
        
    df = pd.DataFrame(rows)
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="Metric", y="Score", hue="Explainer", data=df)
    # sns.violinplot(x="Metric", y="Score", hue="Explainer", data=df, split=True)
    plt.title(title)
    plt.xlabel("Evaluation Metric")
    plt.ylabel("Score")
    plt.legend(title="Explainer", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


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


def visualize_results(res_explainer):
    # make mean and std for each explainer as a column
    for explainer, evals in res_explainer.items():
        for eval_name, scores in evals.items():
            mean = scores['mean']
            std = scores['std']
            col_res = f'%.3f/%.3f' % (mean, std)    
            res_explainer[explainer][eval_name] = col_res
    # show the results using pandas for correct results
    pd.set_option('display.precision', 4)
    df = pd.DataFrame(res_explainer)
    print(df.transpose())

def visualize_split_violin(cor_res_explainer, inc_res_explainer, title, save_path=None):
    """
    Create a split violin plot for correct and incorrect results.
    The inputs cor_res_explainer and inc_res_explainer are dictionaries of the form:
      { Explainer: { Metric: [score, score, ...], ... }, ... }
    
    The function produces a single figure with one facet per Explainer, where for each evaluation
    metric the distribution of scores is shown as a split violin (split by ResultType 'Correct' vs 'Incorrect').
    """
    rows = []
    # Process correct results
    for explainer, evals in cor_res_explainer.items():
        for metric, scores in evals.items():
            # If scores has already been aggregated (mean/std), skip it.
            if isinstance(scores, dict):
                continue
            for s in scores:
                rows.append({
                    "Explainer": explainer,
                    "Metric": metric,
                    "Score": s,
                    "ResultType": "Correct"
                })
    # Process incorrect results
    for explainer, evals in inc_res_explainer.items():
        for metric, scores in evals.items():
            # If scores has already been aggregated (mean/std), skip it.
            if isinstance(scores, dict):
                continue
            for s in scores:
                rows.append({
                    "Explainer": explainer,
                    "Metric": metric,
                    "Score": s,
                    "ResultType": "Incorrect"
                })

    import pandas as pd
    df = pd.DataFrame(rows)

    # Create a facet grid with one subplot per Explainer.
    import seaborn as sns
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    g = sns.catplot(
        x="Metric",
        y="Score",
        hue="ResultType",
        col="Explainer",
        data=df,
        kind="violin",
        split=True,
        height=4,
        aspect=0.8,
        inner="quartile",  # draw quartiles inside
        palette="Set2"
    )
    g.fig.suptitle(title, y=1.05)
    g.set_axis_labels("Evaluation Metric", "Score")
    g._legend.set_title("Result Type")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

# Example usage:
# After you collect your correct and incorrect dictionaries, call:
# visualize_split_violin(cor_res_explainer, inc_res_explainer, "Correct vs. Incorrect Score Distributions")

def analyze_file(file_path):
    # load the pickle file
    # file_path = args.file_path
    data = read_pickle(file_path)

    # analyze the data
    # aggregate the list of data

    cor_res_explainer = {}
    inc_res_explainer = {}

    all_res_explainer = {}

    print(len(data))

    for res in data:
        # print(res.keys())
        correct_results = res['correct_results']
        cor_res_explainer = collect_results(correct_results, cor_res_explainer)

        incorrect_results = res['incorrect_results']
        inc_res_explainer = collect_results(incorrect_results, inc_res_explainer)
        all_results = correct_results + incorrect_results
        # print("all_results: ", all_results, len(all_results))
        all_res_explainer = collect_results(all_results, all_res_explainer)

    # visualize_split_violin(cor_res_explainer, inc_res_explainer, "Correct vs. Incorrect Score Distributions")

    # boxplot for each correct_results
    title = "Correct Results"
    # visualize_boxplot(cor_res_explainer, title)

    cor_res_explainer_dist = copy.deepcopy(cor_res_explainer)
    
    # print(correct_res_explainer)
    for explainer, evals in cor_res_explainer.items():
        # print(f"Explainer: {explainer}")
        for eval_name, scores in evals.items():
            # print(len(scores))
            cor_res_explainer[explainer][eval_name] = {
                "mean": np.mean(scores),
                "std": np.std(scores)
            }

    print("Correct Results")
    print("len(cor_res_explainer): ", len(cor_res_explainer))
    # print("correct_results: ", cor_res_explainer)
    visualize_results(cor_res_explainer)


    # boxplot for each incorrect_results
    title = "Incorrect Results"
    # visualize_boxplot(inc_res_explainer, title)
    
    
    for explainer, evals in inc_res_explainer.items():
        # print(f"Explainer: {explainer}")
        for eval_name, scores in evals.items():
            # print(len(scores))
            inc_res_explainer[explainer][eval_name] = {
                "mean": np.mean(scores),
                "std": np.std(scores)
            }
    
    # print("incorrect_results", inc_res_explainer)
    print("Incorrect Results")
    visualize_results(inc_res_explainer)

    # boxplot for all results
    title = "All Results"
    # visualize_boxplot(all_res_explainer, title)

    all_res_explainer_dist = copy.deepcopy(all_res_explainer)

    for explainer, evals in all_res_explainer.items():
        # print(f"Explainer: {explainer}")
        for eval_name, scores in evals.items():
            # print(eval_name, len(scores))
            all_res_explainer[explainer][eval_name] = {
                "mean": np.mean(scores),
                "std": np.std(scores)
            }

    # print("all_results", all_res_explainer)
    print("All Results")
    visualize_results(all_res_explainer)

    return all_res_explainer_dist
    # return cor_res_explainer_dist


if __name__ == '__main__':
    args = parse_args()
    # analyze_file(args)
    if len(args.file_path) > 1:
        models = {}
        print(args.file_path)
        for file_path in args.file_path:
            # get model name from folder name before the file
            model_name = file_path.split('/')[-2]
            print("Model Name: ", model_name)
            all_res_explainer_dist = analyze_file(file_path)
            models[model_name] = all_res_explainer_dist

        print(len(models))
        # Combine the raw results from all models into one DataFrame for the selected metrics.
        # We assume that models is a dictionary where each key is a Model name and each value is
        # a dictionary of raw distributions (all_res_explainer_dist) with the following structure:
        #   { Explainer: { Metric: [score, score, ...], ... }, ... }
        selected_metrics = ["aopc_compr", "aopc_suff"]
        combined_rows = []
        for model, res in models.items():
            for explainer, evals in res.items():
                for metric, scores in evals.items():
                    if metric in selected_metrics:
                        for s in scores:
                            combined_rows.append({
                                "Model": model,
                                "Explainer": explainer,
                                "Metric": metric,
                                "Score": s
                            })

        combined_df = pd.DataFrame(combined_rows)

        # For each selected metric, create a separate figure.
        for metric in selected_metrics:
            # Filter the DataFrame for the current metric.
            df_metric = combined_df[combined_df["Metric"] == metric]
            
            # Create a new figure for this metric.
            plt.figure(figsize=(8, 6))
            
            # Plot boxplot comparing Model (x-axis) with Score, colored by Explainer.
            ax = sns.boxplot(x="Model", y="Score", hue="Explainer", data=df_metric)
            # ax = sns.violinplot(x="Model", y="Score", hue="Explainer", data=df_metric, split=False)
            ax.set_title(metric)
            ax.set_xlabel("Model")
            ax.set_ylabel("Score")
            # ax.legend(title="Explainer", bbox_to_anchor=(1.05, 1), loc="upper left")
            # change the legend position to the bottom middle
            ax.legend(title="Explainer", bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2)
            
            plt.tight_layout()
            plt.show()

    else:
        file_path = args.file_path[0]
        model_name = file_path.split('/')[-2]
        print("Model Name: ", model_name)
        analyze_file(file_path)
        


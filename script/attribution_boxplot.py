import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main(args):
    # Read the CSV data
    # data = pd.read_csv('attribution.csv')
    data = pd.read_csv(args.input_csv, delimiter=args.delimiter)

    # List of metrics to plot
    metrics = [
        ('aopc_compr_mean', 'aopc_compr_std'),
        ('aopc_suff_mean', 'aopc_suff_std'),
        # ('taucorr_loo_mean', 'taucorr_loo_std')
    ]

    # Generate a bar plot with error bars for each metric
    for mean_col, std_col in metrics:
        plt.figure(figsize=(12, 8))
        
        # Create a bar plot with error bars
        models = data['Model'].unique()
        methods = data['Method'].unique()
        bar_width = 0.2
        index = np.arange(len(models))
        
        for i, method in enumerate(methods):
            method_data = data[data['Method'] == method]
            plt.bar(index + i * bar_width, method_data[mean_col], bar_width, 
                    yerr=method_data[std_col], 
                    label=method, capsize=5)
        
        # Customize the plot
        plt.title(f'Bar Plot of {mean_col.replace("_mean", "").replace("_", " ").title()} by Model and Method')
        plt.xlabel('Model')
        plt.ylabel(mean_col.replace("_mean", "").replace("_", " ").title())
        plt.xticks(index + bar_width * (len(methods) - 1) / 2, models)
        plt.legend(title='Method', bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=int(len(methods)/2))
        
        # Save the plot as a PNG file
        plt.tight_layout()
        # plt.savefig(f'{mean_col}.png')
        plt.savefig(f'{args.output_dir}/{mean_col}.png')
        
        # Show the plot (optional)
        # plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--delimiter", type=str, default=",")
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    main(args)

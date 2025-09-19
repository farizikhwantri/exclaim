import pandas as pd
import numpy as np
from scipy import stats

def permutation_test(group1, group2, n_permutations=10000):
    """
    Perform a permutation test to determine if the difference in means 
    between two groups is statistically significant.
    
    Returns:
    - observed_diff: The actual difference in means
    - p_value: The p-value from the permutation test
    """
    # Calculate observed difference
    observed_diff = np.mean(group1) - np.mean(group2)
    
    # Combine all data
    combined = np.concatenate([group1, group2])
    n1 = len(group1)
    
    # Generate null distribution through permutation
    null_diffs = []
    
    for _ in range(n_permutations):
        # Randomly shuffle and split
        np.random.shuffle(combined)
        perm_group1 = combined[:n1]
        perm_group2 = combined[n1:]
        
        # Calculate difference for this permutation
        perm_diff = np.mean(perm_group1) - np.mean(perm_group2)
        null_diffs.append(perm_diff)
    
    # Calculate p-value (two-tailed test)
    null_diffs = np.array(null_diffs)
    p_value = np.mean(np.abs(null_diffs) >= np.abs(observed_diff))
    
    return observed_diff, p_value, null_diffs

def analyze_aopc_with_significance_testing(csv_file_path, n_permutations=10000):
    """
    Calculate average AOPC comprehensiveness and sufficiency divided by prediction correctness
    with permutation tests for statistical significance.
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Read the CSV file
    df = pd.read_csv(csv_file_path)
    
    print("=== Dataset Overview ===")
    print(f"Total instances: {len(df)}")
    
    # Split data by prediction correctness
    correct_predictions = df[df['predicted_label'] == df['true_label']]
    incorrect_predictions = df[df['predicted_label'] != df['true_label']]
    
    print(f"Correct predictions: {len(correct_predictions)}")
    print(f"Incorrect predictions: {len(incorrect_predictions)}")
    print(f"Overall accuracy: {len(correct_predictions)/len(df):.4f}")

    # calculate overall comprehensiveness and sufficiency regardless correct or incorrect
    overall_avg_comp = df['average_comprehensiveness'].mean()
    print(f"Overall Average Comprehensiveness: {overall_avg_comp:.4f}")
    overall_avg_suff = df['average_sufficiency'].mean()
    print(f"Overall Average Sufficiency: {overall_avg_suff:.4f}")

    # Calculate averages for correct predictions
    if len(correct_predictions) > 0:
        correct_avg_comp = correct_predictions['average_comprehensiveness'].mean()
        correct_avg_suff = correct_predictions['average_sufficiency'].mean()
    else:
        correct_avg_comp = 0
        correct_avg_suff = 0
    
    # Calculate averages for incorrect predictions  
    if len(incorrect_predictions) > 0:
        incorrect_avg_comp = incorrect_predictions['average_comprehensiveness'].mean()
        incorrect_avg_suff = incorrect_predictions['average_sufficiency'].mean()
    else:
        incorrect_avg_comp = 0
        incorrect_avg_suff = 0
    
    print("\n=== AOPC Metrics by Prediction Correctness ===")
    print("\nCorrect Predictions:")
    print(f"  Count: {len(correct_predictions)}")
    print(f"  Average Comprehensiveness: {correct_avg_comp:.4f}")
    print(f"  Average Sufficiency: {correct_avg_suff:.4f}")
    
    print("\nIncorrect Predictions:")
    print(f"  Count: {len(incorrect_predictions)}")
    print(f"  Average Comprehensiveness: {incorrect_avg_comp:.4f}")
    print(f"  Average Sufficiency: {incorrect_avg_suff:.4f}")
    
    # Calculate differences
    if len(correct_predictions) > 0 and len(incorrect_predictions) > 0:
        comp_diff = correct_avg_comp - incorrect_avg_comp
        suff_diff = correct_avg_suff - incorrect_avg_suff
        
        print("\n=== Differences (Correct - Incorrect) ===")
        print(f"Comprehensiveness difference: {comp_diff:.4f}")
        print(f"Sufficiency difference: {suff_diff:.4f}")
        
        # Perform permutation tests
        print(f"\n=== Statistical Significance Testing (n_permutations={n_permutations}) ===")
        
        # Comprehensiveness permutation test
        comp_correct = correct_predictions['average_comprehensiveness'].values
        comp_incorrect = incorrect_predictions['average_comprehensiveness'].values
        
        comp_obs_diff, comp_p_value, comp_null_diffs = permutation_test(
            comp_correct, comp_incorrect, n_permutations
        )
        
        print("\nComprehensiveness Analysis:")
        print(f"  Observed difference: {comp_obs_diff:.4f}")
        print(f"  P-value: {comp_p_value:.4f}")
        print(f"  95% CI of null distribution: [{np.percentile(comp_null_diffs, 2.5):.4f}, {np.percentile(comp_null_diffs, 97.5):.4f}]")
        if comp_p_value < 0.05:
            print(f"  → SIGNIFICANT at α=0.05")
        else:
            print(f"  → NOT significant at α=0.05")
        
        # Sufficiency permutation test
        suff_correct = correct_predictions['average_sufficiency'].values
        suff_incorrect = incorrect_predictions['average_sufficiency'].values
        
        suff_obs_diff, suff_p_value, suff_null_diffs = permutation_test(
            suff_correct, suff_incorrect, n_permutations
        )
        
        print("\nSufficiency Analysis:")
        print(f"  Observed difference: {suff_obs_diff:.4f}")
        print(f"  P-value: {suff_p_value:.4f}")
        print(f"  95% CI of null distribution: [{np.percentile(suff_null_diffs, 2.5):.4f}, {np.percentile(suff_null_diffs, 97.5):.4f}]")
        if suff_p_value < 0.05:
            print(f"  → SIGNIFICANT at α=0.05")
        else:
            print(f"  → NOT significant at α=0.05")
        
        # Additional statistical tests for comparison
        print("\n=== Additional Statistical Tests ===")
        
        # Welch's t-test (unequal variances)
        comp_t_stat, comp_t_p = stats.ttest_ind(comp_correct, comp_incorrect, equal_var=False)
        suff_t_stat, suff_t_p = stats.ttest_ind(suff_correct, suff_incorrect, equal_var=False)
        
        print(f"\nWelch's t-test (unequal variances):")
        print(f"  Comprehensiveness: t={comp_t_stat:.4f}, p={comp_t_p:.4f}")
        print(f"  Sufficiency: t={suff_t_stat:.4f}, p={suff_t_p:.4f}")
        
        # Mann-Whitney U test (non-parametric)
        comp_u_stat, comp_u_p = stats.mannwhitneyu(comp_correct, comp_incorrect, alternative='two-sided')
        suff_u_stat, suff_u_p = stats.mannwhitneyu(suff_correct, suff_incorrect, alternative='two-sided')
        
        print(f"\nMann-Whitney U test (non-parametric):")
        print(f"  Comprehensiveness: U={comp_u_stat:.0f}, p={comp_u_p:.4f}")
        print(f"  Sufficiency: U={suff_u_stat:.0f}, p={suff_u_p:.4f}")
        
        # Effect size (Cohen's d)
        def cohens_d(group1, group2):
            n1, n2 = len(group1), len(group2)
            pooled_std = np.sqrt(((n1-1)*np.var(group1, ddof=1) + (n2-1)*np.var(group2, ddof=1)) / (n1+n2-2))
            return (np.mean(group1) - np.mean(group2)) / pooled_std
        
        comp_cohen_d = cohens_d(comp_correct, comp_incorrect)
        suff_cohen_d = cohens_d(suff_correct, suff_incorrect)
        
        print(f"\nEffect Size (Cohen's d):")
        print(f"  Comprehensiveness: d={comp_cohen_d:.4f}")
        print(f"  Sufficiency: d={suff_cohen_d:.4f}")
        
        # Interpretation of effect size
        def interpret_cohens_d(d):
            abs_d = abs(d)
            if abs_d < 0.2:
                return "negligible"
            elif abs_d < 0.5:
                return "small"
            elif abs_d < 0.8:
                return "medium"
            else:
                return "large"
        
        print(f"  Comprehensiveness effect size: {interpret_cohens_d(comp_cohen_d)}")
        print(f"  Sufficiency effect size: {interpret_cohens_d(suff_cohen_d)}")
    
    # Additional statistics
    print("\n=== Detailed Statistics ===")
    
    if len(correct_predictions) > 0:
        print("\nCorrect Predictions - Comprehensiveness:")
        print(f"  Min: {correct_predictions['average_comprehensiveness'].min():.4f}")
        print(f"  Max: {correct_predictions['average_comprehensiveness'].max():.4f}")
        print(f"  Std: {correct_predictions['average_comprehensiveness'].std():.4f}")
        
        print("\nCorrect Predictions - Sufficiency:")
        print(f"  Min: {correct_predictions['average_sufficiency'].min():.4f}")
        print(f"  Max: {correct_predictions['average_sufficiency'].max():.4f}")
        print(f"  Std: {correct_predictions['average_sufficiency'].std():.4f}")
    
    if len(incorrect_predictions) > 0:
        print("\nIncorrect Predictions - Comprehensiveness:")
        print(f"  Min: {incorrect_predictions['average_comprehensiveness'].min():.4f}")
        print(f"  Max: {incorrect_predictions['average_comprehensiveness'].max():.4f}")
        print(f"  Std: {incorrect_predictions['average_comprehensiveness'].std():.4f}")
        
        print("\nIncorrect Predictions - Sufficiency:")
        print(f"  Min: {incorrect_predictions['average_sufficiency'].min():.4f}")
        print(f"  Max: {incorrect_predictions['average_sufficiency'].max():.4f}")
        print(f"  Std: {incorrect_predictions['average_sufficiency'].std():.4f}")
    
    # Return results as dictionary
    results = {
        'correct_count': len(correct_predictions),
        'incorrect_count': len(incorrect_predictions),
        'correct_avg_comprehensiveness': correct_avg_comp,
        'correct_avg_sufficiency': correct_avg_suff,
        'incorrect_avg_comprehensiveness': incorrect_avg_comp,
        'incorrect_avg_sufficiency': incorrect_avg_suff,
        'overall_accuracy': len(correct_predictions)/len(df)
    }
    
    if len(correct_predictions) > 0 and len(incorrect_predictions) > 0:
        results.update({
            'comprehensiveness_diff': comp_diff,
            'sufficiency_diff': suff_diff,
            'comprehensiveness_permutation_p': comp_p_value,
            'sufficiency_permutation_p': suff_p_value,
            'comprehensiveness_cohens_d': comp_cohen_d,
            'sufficiency_cohens_d': suff_cohen_d,
            'comprehensiveness_t_test_p': comp_t_p,
            'sufficiency_t_test_p': suff_t_p,
            'comprehensiveness_mannwhitney_p': comp_u_p,
            'sufficiency_mannwhitney_p': suff_u_p
        })
    
    return results

def quick_significance_test(csv_file_path, n_permutations=5000):
    """
    Quick version with fewer permutations for faster results
    """
    return analyze_aopc_with_significance_testing(csv_file_path, n_permutations)

# Usage
if __name__ == "__main__":
    # csv_file_path = "predictions.csv"  # Update with your file path
    import sys
    csv_file_path = sys.argv[1]

    # Full analysis with statistical testing
    print("\nAnalyzing:", csv_file_path)
    results = analyze_aopc_with_significance_testing(csv_file_path, n_permutations=10000)
    
    # Quick version for faster results
    # results = quick_significance_test(csv_file_path, n_permutations=5000)
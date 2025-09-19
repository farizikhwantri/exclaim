import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, classification_report
import os

def align_predictions_with_original(predictions_file, original_file):
    """
    Align predictions file with original dataset file by line number
    """
    # Read both files
    pred_df = pd.read_csv(predictions_file)
    orig_df = pd.read_csv(original_file)
    
    print(f"Predictions file: {len(pred_df)} rows")
    print(f"Original file: {len(orig_df)} rows")
    
    # Find minimum length to ensure alignment
    min_length = min(len(pred_df), len(orig_df))
    print(f"Aligning to {min_length} rows")
    
    # Truncate both to same length and reset index
    pred_aligned = pred_df.iloc[:min_length].reset_index(drop=True)
    orig_aligned = orig_df.iloc[:min_length].reset_index(drop=True)
    
    # Combine the dataframes
    combined_df = orig_aligned.copy()
    
    # Add prediction columns from predictions file
    pred_col = 'prediction' if 'prediction' in pred_df.columns else 'predicted_label'
    combined_df['predicted_label'] = pred_aligned[pred_col]
    
    # Ensure we have the true label column
    if 'label' not in combined_df.columns and 'true_label' in pred_aligned.columns:
        combined_df['true_label'] = pred_aligned['true_label']
    elif 'label' in combined_df.columns:
        combined_df['true_label'] = combined_df['label']
    
    return combined_df

def calculate_metrics_by_group(df, group_col):
    """
    Calculate F1, precision, recall, and accuracy by group
    """
    if group_col not in df.columns:
        print(f"Warning: Column '{group_col}' not found in dataset")
        return {}
    
    results = {}
    
    for group_value in df[group_col].unique():
        if pd.isna(group_value):
            continue
            
        group_data = df[df[group_col] == group_value]
        
        if len(group_data) == 0:
            continue
        
        y_true = group_data['true_label']
        y_pred = group_data['predicted_label']
        
        # Calculate metrics
        try:
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            accuracy = accuracy_score(y_true, y_pred)
            
            results[group_value] = {
                'f1_score': f1,
                'precision': precision,
                'recall': recall,
                'accuracy': accuracy,
                'count': len(group_data),
                'correct': (y_true == y_pred).sum()
            }
        except Exception as e:
            print(f"Error calculating metrics for {group_value}: {e}")
            
    return results

def compare_predictions_with_original(*files, group_columns=['hop', 'target', 'model_name', 'requirement']):
    """
    Main function to compare predictions with original file and calculate metrics by groups
    """
    if len(files) < 2:
        raise ValueError("Need at least 2 files: predictions file and original file")
    
    predictions_file = files[0]
    original_file = files[1]
    
    print(f"Comparing {predictions_file} with {original_file}")
    
    # Align the files
    combined_df = align_predictions_with_original(predictions_file, original_file)
    
    print(f"\nDataset columns: {list(combined_df.columns)}")
    print(f"Aligned dataset shape: {combined_df.shape}")
    
    # Overall metrics
    overall_accuracy = (combined_df['predicted_label'] == combined_df['true_label']).mean()
    overall_f1 = f1_score(combined_df['true_label'], combined_df['predicted_label'], average='weighted', zero_division=0)
    
    print(f"\n=== OVERALL METRICS ===")
    print(f"Overall Accuracy: {overall_accuracy:.4f}")
    print(f"Overall F1 Score: {overall_f1:.4f}")
    
    # Calculate metrics for each grouping column
    for group_col in group_columns:
        if group_col in combined_df.columns:
            print(f"\n{'='*50}")
            print(f"ANALYSIS BY {group_col.upper()}")
            print(f"{'='*50}")
            
            results = calculate_metrics_by_group(combined_df, group_col)
            
            if not results:
                print(f"No results for {group_col}")
                continue
            
            # Print results
            for group_value, metrics in sorted(results.items()):
                print(f"\n{group_col} = {group_value} (n={metrics['count']}):")
                print(f"  Accuracy: {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['count']})")
                print(f"  F1 Score: {metrics['f1_score']:.4f}")
                print(f"  Precision: {metrics['precision']:.4f}")
                print(f"  Recall: {metrics['recall']:.4f}")
            
            # Create summary table
            print(f"\n=== {group_col.upper()} SUMMARY TABLE ===")
            print(f"{'Group':<15} {'Count':<8} {'Accuracy':<10} {'F1 Score':<10} {'Precision':<12} {'Recall':<10}")
            print("="*70)
            
            for group_value, metrics in sorted(results.items()):
                print(f"{str(group_value):<15} {metrics['count']:<8} {metrics['accuracy']:<10.4f} "
                      f"{metrics['f1_score']:<10.4f} {metrics['precision']:<12.4f} {metrics['recall']:<10.4f}")
            
            # Save results to CSV
            summary_data = []
            for group_value, metrics in results.items():
                summary_data.append({
                    'group_column': group_col,
                    'group_value': group_value,
                    'count': metrics['count'],
                    'correct': metrics['correct'],
                    'accuracy': metrics['accuracy'],
                    'f1_score': metrics['f1_score'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall']
                })
            
            summary_df = pd.DataFrame(summary_data)
            output_file = f"metrics_by_{group_col}.csv"
            # summary_df.to_csv(output_file, index=False)
            print(f"\nResults saved to {output_file}")
        else:
            print(f"\nColumn '{group_col}' not found in dataset")
    
    # Detailed classification report
    print(f"\n=== DETAILED CLASSIFICATION REPORT ===")
    print(classification_report(combined_df['true_label'], combined_df['predicted_label']))
    
    return combined_df

def analyze_cross_group_performance(df, group_cols=['hop', 'target']):
    """
    Analyze performance across multiple grouping dimensions
    """
    if not all(col in df.columns for col in group_cols):
        missing = [col for col in group_cols if col not in df.columns]
        print(f"Missing columns: {missing}")
        return
    
    print(f"\n=== CROSS-GROUP ANALYSIS ({' x '.join(group_cols)}) ===")
    
    # Group by multiple columns
    grouped = df.groupby(group_cols)
    
    results = []
    for group_keys, group_data in grouped:
        if len(group_data) == 0:
            continue
            
        y_true = group_data['true_label']
        y_pred = group_data['predicted_label']
        
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        result = {
            'count': len(group_data),
            'accuracy': accuracy,
            'f1_score': f1
        }
        
        # Add group keys
        for i, col in enumerate(group_cols):
            result[col] = group_keys[i] if isinstance(group_keys, tuple) else group_keys
        
        results.append(result)
    
    # Create and display results
    results_df = pd.DataFrame(results)
    
    if not results_df.empty:
        print(results_df.to_string(index=False, float_format='%.4f'))
        
        # Save cross-group results
        output_file = f"metrics_cross_{'_'.join(group_cols)}.csv"
        # results_df.to_csv(output_file, index=False)
        print(f"\nCross-group results saved to {output_file}")
    
    return results_df

# Usage example
if __name__ == "__main__":
    # Example usage
    import sys
    predictions_file = sys.argv[1] if len(sys.argv) > 1 else "predictions.csv"
    original_file = sys.argv[2] if len(sys.argv) > 2 else "original_dataset.csv"

    # Run the comparison
    combined_data = compare_predictions_with_original(
        predictions_file, 
        original_file,
        group_columns=['hop']
    )
    
    # Optional: Cross-group analysis
    # if len(combined_data) > 0:
    #     analyze_cross_group_performance(combined_data, ['hop', 'target'])
    #     analyze_cross_group_performance(combined_data, ['hop', 'model_name'])
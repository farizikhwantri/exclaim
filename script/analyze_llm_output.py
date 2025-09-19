import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, classification_report
import numpy as np

def calculate_f2_score(precision, recall):
    """Calculate F2 score manually"""
    if precision + recall == 0:
        return 0
    return (1 + 2**2) * (precision * recall) / ((2**2 * precision) + recall)

def analyze_predictions(csv_file_path):
    # Read the CSV file
    df = pd.read_csv(csv_file_path)
    
    # Get predictions and true labels
    y_true = df['label']
    y_pred = df['prediction']
    
    # Get unique labels
    labels = sorted(df['label'].unique())
    print(f"Unique labels: {labels}")
    
    # Calculate overall metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
    )
    
    # Calculate F2 score
    f2 = calculate_f2_score(precision, recall)
    
    print("\n=== OVERALL METRICS (Weighted Average) ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"F2 Score: {f2:.4f}")
    
    # Calculate metrics for each class
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=labels
    )
    
    print("\n=== PER-CLASS METRICS ===")
    for i, label in enumerate(labels):
        f2_class = calculate_f2_score(precision_per_class[i], recall_per_class[i])
        print(f"\nClass: {label}")
        print(f"  Precision: {precision_per_class[i]:.4f}")
        print(f"  Recall: {recall_per_class[i]:.4f}")
        print(f"  F1 Score: {f1_per_class[i]:.4f}")
        print(f"  F2 Score: {f2_class:.4f}")
        print(f"  Support: {support_per_class[i]}")
    
    # Macro averages
    macro_precision = np.mean(precision_per_class)
    macro_recall = np.mean(recall_per_class)
    macro_f1 = np.mean(f1_per_class)
    macro_f2 = calculate_f2_score(macro_precision, macro_recall)
    
    print("\n=== MACRO AVERAGES ===")
    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Macro Recall: {macro_recall:.4f}")
    print(f"Macro F1 Score: {macro_f1:.4f}")
    print(f"Macro F2 Score: {macro_f2:.4f}")
    
    # Detailed classification report
    print("\n=== DETAILED CLASSIFICATION REPORT ===")
    print(classification_report(y_true, y_pred))
    
    # Confusion matrix information
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("\n=== CONFUSION MATRIX ===")
    print(f"Labels: {labels}")
    print(cm)
    
    # Calculate accuracy
    accuracy = (y_true == y_pred).mean()
    print(f"\n=== ACCURACY ===")
    print(f"Accuracy: {accuracy:.4f}")
    
    # Distribution of predictions vs actual
    print("\n=== LABEL DISTRIBUTION ===")
    print("Actual labels:")
    print(y_true.value_counts().sort_index())
    print("\nPredicted labels:")
    print(y_pred.value_counts().sort_index())

# Run the analysis
if __name__ == "__main__":
    import sys
    # Replace with your CSV file path
    csv_file_path = sys.argv[1]
    analyze_predictions(csv_file_path)
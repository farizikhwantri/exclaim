
def f2_score(precision, recall):
    """
    Compute the F2 score given precision and recall.
    """
    if precision + recall == 0:
        return 0.0
    return (5 * precision * recall) / (4 * precision + recall)

def calculate_metrics(results):
    """
    Calculate accuracy metrics from the results
    """
    labels = []
    scores = []
    predictions = []

    for result in results:
        labels.append(result["label"])
        scores.append(result["output"])  # Probability score (0.0-1.0)
        predictions.append(result["predicted_score"])  # Binary prediction (0/1)
    total_images = len(labels)

    # Calculate metrics
    true_pos = sum(1 for l, p in zip(labels, predictions) if l == 1 and p == 1)
    true_neg = sum(1 for l, p in zip(labels, predictions) if l == 0 and p == 0)
    false_pos = sum(1 for l, p in zip(labels, predictions) if l == 0 and p == 1)
    false_neg = sum(1 for l, p in zip(labels, predictions) if l == 1 and p == 0)

    correct = sum(1 for l, p in zip(labels, predictions) if l == p)
    accuracy = correct / total_images if total_images > 0 else 0
    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Distribution of scores
    score_distribution = {
        "0.0": sum(1 for score in scores if score == 0.0),
        "0.2": sum(1 for score in scores if score == 0.2),
        "0.4": sum(1 for score in scores if score == 0.4),
        "0.6": sum(1 for score in scores if score == 0.6),
        "0.8": sum(1 for score in scores if score == 0.8),
        "1.0": sum(1 for score in scores if score == 1.0),
    }

    return {
        "total_images": total_images,
        "score_distribution": score_distribution,
        "true_positive": true_pos,
        "true_negative": true_neg,
        "false_positive": false_pos,
        "false_negative": false_neg,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

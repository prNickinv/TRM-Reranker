"""Metrics for reranking evaluation."""

import torch
import numpy as np
from typing import List, Dict


def compute_mrr(predictions: torch.Tensor, labels: torch.Tensor, k: int = 10) -> float:
    """
    Compute Mean Reciprocal Rank @k.
    
    Args:
        predictions: [batch_size, num_docs] relevance scores
        labels: [batch_size, num_docs] binary relevance labels
        k: cutoff rank
    """
    # Get top-k predictions
    _, indices = torch.topk(predictions, k=min(k, predictions.size(1)), dim=1)
    
    # Gather labels for top-k
    batch_size = predictions.size(0)
    top_k_labels = torch.gather(labels, 1, indices)
    
    # Find first relevant document position
    reciprocal_ranks = []
    for i in range(batch_size):
        relevant_positions = (top_k_labels[i] > 0).nonzero(as_tuple=True)[0]
        if len(relevant_positions) > 0:
            rr = 1.0 / (relevant_positions[0].item() + 1)
        else:
            rr = 0.0
        reciprocal_ranks.append(rr)
    
    return float(np.mean(reciprocal_ranks))


def compute_ndcg(predictions: torch.Tensor, labels: torch.Tensor, k: int = 10) -> float:
    """
    Compute Normalized Discounted Cumulative Gain @k.
    
    Args:
        predictions: [batch_size, num_docs] relevance scores
        labels: [batch_size, num_docs] relevance labels (can be graded)
        k: cutoff rank
    """
    def dcg_at_k(relevances, k):
        relevances = relevances[:k]
        if relevances.size == 0:
            return 0.0
        discounts = np.log2(np.arange(2, relevances.size + 2))
        return np.sum(relevances / discounts)
    
    ndcg_scores = []
    predictions_np = predictions.cpu().numpy()
    labels_np = labels.cpu().numpy()
    
    for pred, label in zip(predictions_np, labels_np):
        # Get predicted ranking
        pred_order = np.argsort(-pred)
        pred_relevances = label[pred_order]
        
        # Get ideal ranking
        ideal_order = np.argsort(-label)
        ideal_relevances = label[ideal_order]
        
        # Compute DCG and IDCG
        dcg = dcg_at_k(pred_relevances, k)
        idcg = dcg_at_k(ideal_relevances, k)
        
        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcg_scores.append(ndcg)
    
    return float(np.mean(ndcg_scores))


def compute_recall(predictions: torch.Tensor, labels: torch.Tensor, k: int = 10) -> float:
    """
    Compute Recall@k.
    
    Args:
        predictions: [batch_size, num_docs] relevance scores
        labels: [batch_size, num_docs] binary relevance labels
        k: cutoff rank
    """
    _, indices = torch.topk(predictions, k=min(k, predictions.size(1)), dim=1)
    
    batch_size = predictions.size(0)
    top_k_labels = torch.gather(labels, 1, indices)
    
    recalls = []
    for i in range(batch_size):
        num_relevant = (labels[i] > 0).sum().item()
        if num_relevant == 0:
            continue
        num_relevant_retrieved = (top_k_labels[i] > 0).sum().item()
        recall = num_relevant_retrieved / num_relevant
        recalls.append(recall)
    
    return float(np.mean(recalls)) if recalls else 0.0


def compute_binary_classification_metrics(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute binary classification metrics: accuracy, precision, recall, F1.

    Args:
        predictions: [batch_size] predicted scores (logits or probabilities)
        labels: [batch_size] binary labels (0 or 1)
        threshold: threshold for converting predictions to binary (default: 0.5)

    Returns:
        Dictionary with 'accuracy', 'precision', 'recall', 'f1'
    """
    # Convert predictions to binary using threshold
    if predictions.dim() > 1:
        predictions = predictions.squeeze()
    if labels.dim() > 1:
        labels = labels.squeeze()

    # Apply sigmoid if predictions look like logits (values outside [0, 1])
    if predictions.min() < 0 or predictions.max() > 1:
        preds_binary = (torch.sigmoid(predictions) > threshold).float()
    else:
        preds_binary = (predictions > threshold).float()

    labels = labels.float()

    # True Positives, False Positives, True Negatives, False Negatives
    tp = ((preds_binary == 1) & (labels == 1)).sum().item()
    fp = ((preds_binary == 1) & (labels == 0)).sum().item()
    tn = ((preds_binary == 0) & (labels == 0)).sum().item()
    fn = ((preds_binary == 0) & (labels == 1)).sum().item()

    # Accuracy
    accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0.0

    # Precision
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # Recall
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # F1 Score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'tp': float(tp),
        'fp': float(fp),
        'tn': float(tn),
        'fn': float(fn),
    }


def compute_ranking_metrics(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    k_values: List[int] = [1, 5, 10, 20, 100]
) -> Dict[str, float]:
    """
    Compute all ranking metrics for multiple k values.

    Returns:
        Dictionary with keys like 'MRR@10', 'NDCG@10', 'Recall@10'
    """
    metrics = {}

    for k in k_values:
        metrics[f'MRR@{k}'] = compute_mrr(predictions, labels, k)
        metrics[f'NDCG@{k}'] = compute_ndcg(predictions, labels, k)
        metrics[f'Recall@{k}'] = compute_recall(predictions, labels, k)

    return metrics

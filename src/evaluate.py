"""
Evaluation Metrics for Semantic Product Search
===============================================

Implements ranking metrics:
- NDCG (Normalized Discounted Cumulative Gain)
- MAP (Mean Average Precision)
- Precision@K, Recall@K, F1@K

Author: [Your Name]
Course: Generative AI - Fall 2025
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from sklearn.metrics import ndcg_score
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


class RankingMetrics:
    """
    Calculate ranking evaluation metrics for information retrieval
    """
    
    def __init__(self, k_values: List[int] = [5, 10, 20]):
        """
        Initialize metrics calculator
        
        Args:
            k_values: List of K values for @K metrics
        """
        self.k_values = k_values
    
    def ndcg_at_k(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        k: int
    ) -> float:
        """
        Calculate NDCG@K (Normalized Discounted Cumulative Gain)
        
        Args:
            y_true: True relevance scores
            y_pred: Predicted relevance scores
            k: Number of top results to consider
        
        Returns:
            NDCG@K score
        """
        if len(y_true) == 0:
            return 0.0
        
        # Truncate to top K
        y_true_k = y_true[:k]
        y_pred_k = y_pred[:k]
        
        # Use sklearn's implementation
        try:
            score = ndcg_score([y_true_k], [y_pred_k])
        except:
            score = 0.0
        
        return score
    
    def precision_at_k(
        self,
        relevant_items: set,
        retrieved_items: list,
        k: int
    ) -> float:
        """
        Calculate Precision@K
        
        Precision@K = (# relevant items in top K) / K
        
        Args:
            relevant_items: Set of relevant item IDs
            retrieved_items: List of retrieved item IDs (ordered by score)
            k: Number of top results to consider
        
        Returns:
            Precision@K score
        """
        if k == 0:
            return 0.0
        
        # Get top K items
        top_k = set(retrieved_items[:k])
        
        # Count relevant items in top K
        relevant_in_top_k = len(relevant_items & top_k)
        
        return relevant_in_top_k / k
    
    def recall_at_k(
        self,
        relevant_items: set,
        retrieved_items: list,
        k: int
    ) -> float:
        """
        Calculate Recall@K
        
        Recall@K = (# relevant items in top K) / (total # relevant items)
        
        Args:
            relevant_items: Set of relevant item IDs
            retrieved_items: List of retrieved item IDs (ordered by score)
            k: Number of top results to consider
        
        Returns:
            Recall@K score
        """
        if len(relevant_items) == 0:
            return 0.0
        
        # Get top K items
        top_k = set(retrieved_items[:k])
        
        # Count relevant items in top K
        relevant_in_top_k = len(relevant_items & top_k)
        
        return relevant_in_top_k / len(relevant_items)
    
    def f1_at_k(
        self,
        relevant_items: set,
        retrieved_items: list,
        k: int
    ) -> float:
        """
        Calculate F1@K
        
        F1@K = 2 * (Precision@K * Recall@K) / (Precision@K + Recall@K)
        
        Args:
            relevant_items: Set of relevant item IDs
            retrieved_items: List of retrieved item IDs
            k: Number of top results to consider
        
        Returns:
            F1@K score
        """
        precision = self.precision_at_k(relevant_items, retrieved_items, k)
        recall = self.recall_at_k(relevant_items, retrieved_items, k)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def average_precision(
        self,
        relevant_items: set,
        retrieved_items: list
    ) -> float:
        """
        Calculate Average Precision for a single query
        
        AP = (sum of P@k for each relevant item) / (total relevant items)
        
        Args:
            relevant_items: Set of relevant item IDs
            retrieved_items: List of retrieved item IDs (ordered by score)
        
        Returns:
            Average Precision score
        """
        if len(relevant_items) == 0:
            return 0.0
        
        precisions = []
        num_relevant_seen = 0
        
        for idx, item in enumerate(retrieved_items, 1):
            if item in relevant_items:
                num_relevant_seen += 1
                precision_at_idx = num_relevant_seen / idx
                precisions.append(precision_at_idx)
        
        if len(precisions) == 0:
            return 0.0
        
        return np.mean(precisions)
    
    def evaluate_ranking(
        self,
        df: pd.DataFrame,
        predictions: np.ndarray,
        relevance_threshold: float = 0.0
    ) -> Dict[str, float]:
        """
        Evaluate ranking performance on a dataset
        
        Args:
            df: DataFrame with columns: query_id, product_id, relevance_score
            predictions: Predicted relevance scores (aligned with df)
            relevance_threshold: Threshold for considering item as relevant
        
        Returns:
            Dictionary of metric names to values
        """
        # Add predictions to dataframe
        df = df.copy()
        df['predicted_score'] = predictions
        
        metrics = defaultdict(list)
        
        # Evaluate per query
        for query_id, group in df.groupby('query_id'):
            # Sort by predicted score (descending)
            group_sorted = group.sort_values('predicted_score', ascending=False)
            
            # Get relevant items (ground truth)
            relevant_items = set(
                group[group['relevance_score'] > relevance_threshold]['product_id']
            )
            
            # Get retrieved items (ordered)
            retrieved_items = group_sorted['product_id'].tolist()
            
            # Calculate metrics for each K
            for k in self.k_values:
                metrics[f'precision@{k}'].append(
                    self.precision_at_k(relevant_items, retrieved_items, k)
                )
                metrics[f'recall@{k}'].append(
                    self.recall_at_k(relevant_items, retrieved_items, k)
                )
                metrics[f'f1@{k}'].append(
                    self.f1_at_k(relevant_items, retrieved_items, k)
                )
                
                # NDCG@K
                y_true = group_sorted['relevance_score'].values[:k]
                y_pred = group_sorted['predicted_score'].values[:k]
                
                if len(y_true) > 0:
                    ndcg = self.ndcg_at_k(y_true, y_pred, k)
                    metrics[f'ndcg@{k}'].append(ndcg)
            
            # Average Precision (for MAP)
            ap = self.average_precision(relevant_items, retrieved_items)
            metrics['ap'].append(ap)
        
        # Average across all queries
        result = {}
        for metric_name, values in metrics.items():
            if metric_name == 'ap':
                result['map'] = np.mean(values)
            else:
                result[metric_name] = np.mean(values)
        
        return result
    
    def print_metrics(self, metrics: Dict[str, float]):
        """
        Print metrics in a formatted table
        
        Args:
            metrics: Dictionary of metrics
        """
        print("\n" + "=" * 70)
        print("📊 RANKING EVALUATION METRICS")
        print("=" * 70)
        
        print(f"\n{'Metric':<20} {'@5':<12} {'@10':<12} {'@20':<12}")
        print("-" * 56)
        
        for metric_name in ['ndcg', 'precision', 'recall', 'f1']:
            values = []
            for k in self.k_values:
                key = f'{metric_name}@{k}'
                values.append(metrics.get(key, 0.0))
            
            print(f"{metric_name.upper():<20} {values[0]:<12.4f} {values[1]:<12.4f} {values[2]:<12.4f}")
        
        print("\n" + "-" * 56)
        print(f"MAP (Mean Avg Precision): {metrics.get('map', 0.0):.4f}")
        print("=" * 70 + "\n")


def calculate_esci_accuracy(
    df: pd.DataFrame,
    predictions: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculate accuracy metrics for ESCI label prediction
    
    Args:
        df: DataFrame with esci_label column
        predictions: Predicted relevance scores
        threshold: Threshold for binary classification
    
    Returns:
        Dictionary of accuracy metrics
    """
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    
    # Convert ESCI to binary (relevant vs irrelevant)
    y_true = (df['relevance_score'] > 0).astype(int)
    y_pred = (predictions > threshold).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary'
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


if __name__ == "__main__":
    # Test metrics with synthetic data
    print("Testing Ranking Metrics...")
    
    # Create synthetic query results
    df_test = pd.DataFrame({
        'query_id': ['q1'] * 10 + ['q2'] * 10,
        'product_id': [f'p{i}' for i in range(20)],
        'relevance_score': [1.0, 1.0, 0.7, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                           1.0, 0.7, 0.7, 0.3, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0]
    })
    
    # Simulated predictions (with some errors)
    predictions = np.array([0.9, 0.85, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05,
                           0.95, 0.9, 0.85, 0.7, 0.65, 0.4, 0.35, 0.25, 0.15, 0.1])
    
    # Calculate metrics
    evaluator = RankingMetrics(k_values=[5, 10, 20])
    metrics = evaluator.evaluate_ranking(df_test, predictions)
    
    # Print results
    evaluator.print_metrics(metrics)
    
    # Test ESCI accuracy
    esci_metrics = calculate_esci_accuracy(df_test, predictions)
    print("\n📊 ESCI Classification Metrics:")
    for metric, value in esci_metrics.items():
        print(f"   {metric}: {value:.4f}")
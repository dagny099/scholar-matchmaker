"""
Evaluation metrics for citation link prediction.
Implements Mean Reciprocal Rank (MRR) and Hits@K as described in the README.
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

from .model import TransE, TransETrainer


class LinkPredictionEvaluator:
    """Evaluate TransE model on link prediction task."""
    
    def __init__(self, 
                 model: TransE,
                 entity_mapping: Dict[str, int],
                 device: str = "cpu"):
        self.model = model.to(device)
        self.device = device
        self.entity_mapping = entity_mapping
        self.id_to_entity = {v: k for k, v in entity_mapping.items()}
        
    def mean_reciprocal_rank(self, 
                           test_edges: torch.Tensor,
                           all_entities: Optional[torch.Tensor] = None) -> float:
        """
        Compute Mean Reciprocal Rank (MRR) for test edges.
        
        For each test edge (s, t), rank all possible targets and find 
        the rank of the true target t.
        """
        if all_entities is None:
            all_entities = torch.arange(self.model.num_entities, device=self.device)
        
        self.model.eval()
        reciprocal_ranks = []
        
        with torch.no_grad():
            for source, true_target in tqdm(test_edges, desc="Computing MRR"):
                source = source.to(self.device)
                true_target = true_target.to(self.device)
                
                # Get scores for all possible targets
                scores = self.model.predict_links(
                    source.unsqueeze(0), 
                    all_entities
                ).squeeze(0)  # [num_entities]
                
                # Rank candidates (lower score = better)
                _, ranked_indices = torch.sort(scores)
                
                # Find rank of true target (1-indexed)
                true_rank = (ranked_indices == true_target).nonzero().item() + 1
                reciprocal_ranks.append(1.0 / true_rank)
        
        return np.mean(reciprocal_ranks)
    
    def hits_at_k(self, 
                  test_edges: torch.Tensor,
                  k_values: List[int] = [1, 3, 10],
                  all_entities: Optional[torch.Tensor] = None) -> Dict[int, float]:
        """
        Compute Hits@K for different values of K.
        
        Hits@K = proportion of test cases where true target is in top-K predictions.
        """
        if all_entities is None:
            all_entities = torch.arange(self.model.num_entities, device=self.device)
        
        self.model.eval()
        hits = {k: 0 for k in k_values}
        max_k = max(k_values)
        
        with torch.no_grad():
            for source, true_target in tqdm(test_edges, desc="Computing Hits@K"):
                source = source.to(self.device)
                true_target = true_target.to(self.device)
                
                # Get scores for all possible targets
                scores = self.model.predict_links(
                    source.unsqueeze(0),
                    all_entities
                ).squeeze(0)  # [num_entities]
                
                # Get top-K predictions (lowest scores)
                _, top_k_indices = torch.topk(scores, max_k, largest=False)
                
                # Check if true target is in top-K for each K
                for k in k_values:
                    if true_target in top_k_indices[:k]:
                        hits[k] += 1
        
        # Convert to proportions
        num_test = len(test_edges)
        return {k: hits[k] / num_test for k in k_values}
    
    def binary_classification_metrics(self,
                                    test_edges: torch.Tensor,
                                    test_labels: torch.Tensor) -> Dict[str, float]:
        """
        Evaluate as binary classification (citation exists or not).
        
        Args:
            test_edges: [num_samples, 2] tensor of (source, target) pairs
            test_labels: [num_samples] tensor of binary labels (1=citation exists)
        """
        self.model.eval()
        
        with torch.no_grad():
            test_edges = test_edges.to(self.device)
            scores = self.model(test_edges[:, 0], test_edges[:, 1])
            
            # Convert scores to probabilities (lower score = higher probability)
            probabilities = torch.sigmoid(-scores).cpu().numpy()
            labels = test_labels.cpu().numpy()
        
        # Compute metrics
        auc_score = roc_auc_score(labels, probabilities)
        ap_score = average_precision_score(labels, probabilities)
        
        return {
            "auc": auc_score,
            "average_precision": ap_score
        }
    
    def evaluate_comprehensive(self,
                             test_pos_edges: torch.Tensor,
                             test_neg_edges: torch.Tensor,
                             k_values: List[int] = [1, 3, 10]) -> Dict[str, any]:
        """
        Run comprehensive evaluation including MRR, Hits@K, and binary classification.
        """
        # Combine positive and negative test edges
        test_edges = torch.cat([test_pos_edges, test_neg_edges])
        test_labels = torch.cat([
            torch.ones(len(test_pos_edges)),
            torch.zeros(len(test_neg_edges))
        ])
        
        # Ranking-based metrics (only on positive edges)
        print("Computing ranking metrics...")
        mrr = self.mean_reciprocal_rank(test_pos_edges)
        hits = self.hits_at_k(test_pos_edges, k_values)
        
        # Binary classification metrics
        print("Computing classification metrics...")
        binary_metrics = self.binary_classification_metrics(test_edges, test_labels)
        
        results = {
            "mrr": mrr,
            "hits": hits,
            **binary_metrics
        }
        
        return results
    
    def predict_missing_citations(self,
                                source_papers: List[str],
                                candidate_papers: Optional[List[str]] = None,
                                top_k: int = 10,
                                exclude_existing: bool = True,
                                existing_citations: Optional[Set[Tuple[str, str]]] = None) -> pd.DataFrame:
        """
        Predict missing citations for given source papers.
        
        Args:
            source_papers: Paper IDs to predict citations for
            candidate_papers: Candidate target papers (if None, use all papers)
            top_k: Number of top predictions to return per source
            exclude_existing: Whether to exclude already existing citations
            existing_citations: Set of (source, target) pairs that already exist
        
        Returns:
            DataFrame with columns: source_paper, target_paper, score, rank
        """
        self.model.eval()
        
        # Convert paper IDs to entity indices
        source_indices = [self.entity_mapping[paper] for paper in source_papers 
                         if paper in self.entity_mapping]
        
        if candidate_papers is None:
            candidate_indices = torch.arange(self.model.num_entities)
        else:
            candidate_indices = torch.tensor([self.entity_mapping[paper] 
                                            for paper in candidate_papers 
                                            if paper in self.entity_mapping])
        
        results = []
        
        with torch.no_grad():
            for source_idx in tqdm(source_indices, desc="Predicting citations"):
                source_paper = self.id_to_entity[source_idx]
                
                # Get prediction scores
                scores = self.model.predict_links(
                    torch.tensor([source_idx], device=self.device),
                    candidate_indices.to(self.device)
                ).squeeze(0)  # [num_candidates]
                
                # Sort by score (lower = better)
                sorted_scores, sorted_indices = torch.sort(scores)
                
                # Get top-K predictions
                count = 0
                for rank, (score, candidate_idx) in enumerate(zip(sorted_scores, sorted_indices), 1):
                    candidate_idx = candidate_indices[candidate_idx].item()
                    target_paper = self.id_to_entity[candidate_idx]
                    
                    # Skip self-citations
                    if source_paper == target_paper:
                        continue
                    
                    # Skip existing citations if requested
                    if exclude_existing and existing_citations and (source_paper, target_paper) in existing_citations:
                        continue
                    
                    results.append({
                        "source_paper": source_paper,
                        "target_paper": target_paper,
                        "score": score.item(),
                        "rank": rank
                    })
                    
                    count += 1
                    if count >= top_k:
                        break
        
        return pd.DataFrame(results)


def evaluate_model(trainer: TransETrainer,
                  test_pos_edges: torch.Tensor,
                  test_neg_edges: torch.Tensor,
                  entity_mapping: Dict[str, int],
                  k_values: List[int] = [1, 3, 10]) -> Dict[str, any]:
    """
    Convenience function for comprehensive model evaluation.
    """
    evaluator = LinkPredictionEvaluator(
        trainer.model, 
        entity_mapping, 
        trainer.device
    )
    
    return evaluator.evaluate_comprehensive(
        test_pos_edges, 
        test_neg_edges, 
        k_values
    )
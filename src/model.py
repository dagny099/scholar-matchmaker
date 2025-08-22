"""
TransE model implementation for citation link prediction.
Learning embeddings where cited papers are "close" to citing papers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm


class TransE(nn.Module):
    """
    TransE model for learning paper embeddings.
    
    The model learns embeddings such that for a citation (source, target):
    embedding(source) + relation ≈ embedding(target)
    
    For citation prediction, we use a single "CITES" relation.
    """
    
    def __init__(self, 
                 num_entities: int,
                 embedding_dim: int = 128,
                 margin: float = 1.0,
                 p_norm: int = 1):
        super().__init__()
        
        self.num_entities = num_entities
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.p_norm = p_norm
        
        # Entity embeddings (papers)
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        
        # Relation embedding (we only have "CITES" relation)
        self.relation_embedding = nn.Embedding(1, embedding_dim)
        
        # Initialize embeddings
        self._initialize_embeddings()
        
    def _initialize_embeddings(self):
        """Initialize embeddings with Xavier uniform."""
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embedding.weight)
        
        # Normalize entity embeddings
        with torch.no_grad():
            self.entity_embeddings.weight.div_(
                self.entity_embeddings.weight.norm(p=2, dim=1, keepdim=True)
            )
    
    def forward(self, sources: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: compute scores for (source, target) pairs.
        
        Args:
            sources: Tensor of source entity IDs [batch_size]
            targets: Tensor of target entity IDs [batch_size]
        
        Returns:
            Scores (lower = more likely citation) [batch_size]
        """
        # Get embeddings
        source_emb = self.entity_embeddings(sources)  # [batch_size, embedding_dim]
        target_emb = self.entity_embeddings(targets)  # [batch_size, embedding_dim] 
        relation_emb = self.relation_embedding(torch.zeros_like(sources))  # [batch_size, embedding_dim]
        
        # TransE score: ||source + relation - target||
        score = source_emb + relation_emb - target_emb  # [batch_size, embedding_dim]
        score = torch.norm(score, p=self.p_norm, dim=1)  # [batch_size]
        
        return score
    
    def predict_links(self, sources: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        """
        Predict citation probabilities for source papers to candidate targets.
        
        Args:
            sources: Source paper IDs [num_sources]
            candidates: Candidate target paper IDs [num_candidates]
        
        Returns:
            Scores matrix [num_sources, num_candidates] (lower = more likely)
        """
        sources = sources.unsqueeze(1)  # [num_sources, 1]
        candidates = candidates.unsqueeze(0)  # [1, num_candidates]
        
        # Broadcast to [num_sources, num_candidates]
        sources_expanded = sources.expand(-1, candidates.size(1))
        candidates_expanded = candidates.expand(sources.size(0), -1)
        
        # Flatten for forward pass
        sources_flat = sources_expanded.flatten()
        candidates_flat = candidates_expanded.flatten()
        
        scores = self.forward(sources_flat, candidates_flat)
        return scores.view(sources.size(0), candidates.size(1))


class TransETrainer:
    """Training loop for TransE model."""
    
    def __init__(self, 
                 model: TransE,
                 learning_rate: float = 0.01,
                 device: str = "cpu"):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.loss_history: List[float] = []
        
    def compute_loss(self, 
                    pos_sources: torch.Tensor, 
                    pos_targets: torch.Tensor,
                    neg_sources: torch.Tensor, 
                    neg_targets: torch.Tensor) -> torch.Tensor:
        """Compute margin ranking loss."""
        pos_scores = self.model(pos_sources, pos_targets)
        neg_scores = self.model(neg_sources, neg_targets)
        
        # Margin ranking loss: max(0, margin + pos_score - neg_score)
        loss = F.relu(self.model.margin + pos_scores - neg_scores).mean()
        return loss
    
    def train_epoch(self, 
                   positive_edges: torch.Tensor,
                   negative_edges: torch.Tensor,
                   batch_size: int = 1024) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Ensure we have equal numbers of positive and negative samples
        min_samples = min(len(positive_edges), len(negative_edges))
        pos_edges = positive_edges[:min_samples]
        neg_edges = negative_edges[:min_samples]
        
        # Create batches
        for i in range(0, min_samples, batch_size):
            end_idx = min(i + batch_size, min_samples)
            
            pos_batch = pos_edges[i:end_idx].to(self.device)
            neg_batch = neg_edges[i:end_idx].to(self.device)
            
            pos_sources, pos_targets = pos_batch[:, 0], pos_batch[:, 1]
            neg_sources, neg_targets = neg_batch[:, 0], neg_batch[:, 1]
            
            # Forward pass
            loss = self.compute_loss(pos_sources, pos_targets, neg_sources, neg_targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Normalize embeddings
            with torch.no_grad():
                self.model.entity_embeddings.weight.div_(
                    self.model.entity_embeddings.weight.norm(p=2, dim=1, keepdim=True)
                )
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        self.loss_history.append(avg_loss)
        return avg_loss
    
    def train(self, 
             positive_edges: torch.Tensor,
             negative_edges: torch.Tensor, 
             epochs: int = 100,
             batch_size: int = 1024,
             verbose: bool = True) -> Dict[str, List[float]]:
        """Train the model."""
        if verbose:
            pbar = tqdm(range(epochs), desc="Training TransE")
        else:
            pbar = range(epochs)
            
        for epoch in pbar:
            loss = self.train_epoch(positive_edges, negative_edges, batch_size)
            
            if verbose and epoch % 10 == 0:
                pbar.set_postfix({"Loss": f"{loss:.4f}"})
        
        return {"loss": self.loss_history}
    
    def save_model(self, path: str):
        """Save trained model."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss_history": self.loss_history,
            "model_config": {
                "num_entities": self.model.num_entities,
                "embedding_dim": self.model.embedding_dim,
                "margin": self.model.margin,
                "p_norm": self.model.p_norm
            }
        }, path)
    
    @classmethod
    def load_model(cls, path: str, device: str = "cpu") -> "TransETrainer":
        """Load trained model."""
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint["model_config"]
        
        model = TransE(**config)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        trainer = cls(model, device=device)
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        trainer.loss_history = checkpoint["loss_history"]
        
        return trainer


def create_model(num_entities: int, 
                embedding_dim: int = 128,
                margin: float = 1.0,
                learning_rate: float = 0.01,
                device: str = "cpu") -> TransETrainer:
    """Convenience function to create and initialize a TransE trainer."""
    model = TransE(
        num_entities=num_entities,
        embedding_dim=embedding_dim, 
        margin=margin
    )
    trainer = TransETrainer(model, learning_rate=learning_rate, device=device)
    return trainer
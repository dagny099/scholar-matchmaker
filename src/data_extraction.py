"""
Data extraction and preprocessing pipeline for TransE citation prediction.
Converts Neo4j graph data into training-ready PyTorch tensors.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import networkx as nx
import torch
from typing import Dict, List, Tuple, Set, Optional
from sklearn.model_selection import train_test_split
from .db import Neo4jConnection, QUERIES


class CitationGraphExtractor:
    """Extract and preprocess citation graph data for TransE training."""
    
    def __init__(self, db: Neo4jConnection):
        self.db = db
        self.paper_to_id: Dict[str, int] = {}
        self.id_to_paper: Dict[int, str] = {}
        self.num_entities: int = 0
        self.citation_edges: List[Tuple[int, int]] = []
        self.all_papers_df: pd.DataFrame = pd.DataFrame()
        
    def extract_graph_data(self) -> None:
        """Extract citation graph from Neo4j and build entity mappings."""
        print("Extracting citation edges...")
        edges_df = self.db.query(QUERIES["CITATION_EDGES"])
        
        print("Extracting paper metadata...")  
        self.all_papers_df = self.db.query(QUERIES["ALL_PAPERS"])
        
        # Build entity mappings (paper_id -> integer index)
        unique_papers = set(edges_df["source_id"]) | set(edges_df["target_id"])
        self.paper_to_id = {paper_id: idx for idx, paper_id in enumerate(sorted(unique_papers))}
        self.id_to_paper = {idx: paper_id for paper_id, idx in self.paper_to_id.items()}
        self.num_entities = len(self.paper_to_id)
        
        # Convert to integer edge list
        self.citation_edges = [
            (self.paper_to_id[row["source_id"]], self.paper_to_id[row["target_id"]])
            for _, row in edges_df.iterrows()
            if row["source_id"] in self.paper_to_id and row["target_id"] in self.paper_to_id
        ]
        
        print(f"Extracted {len(self.citation_edges)} citation edges between {self.num_entities} papers")
    
    def build_networkx_graph(self) -> nx.DiGraph:
        """Build NetworkX graph for analysis and visualization."""
        G = nx.DiGraph()
        
        # Add nodes with metadata
        for _, paper in self.all_papers_df.iterrows():
            if paper["paper_id"] in self.paper_to_id:
                node_id = self.paper_to_id[paper["paper_id"]]
                G.add_node(
                    node_id,
                    paper_id=paper["paper_id"],
                    title=paper.get("title", ""),
                    citation_count=paper.get("citation_count", 0),
                    pub_date=paper.get("pub_date", "")
                )
        
        # Add citation edges
        G.add_edges_from(self.citation_edges)
        
        return G
    
    def generate_negative_samples(self, 
                                positive_edges: List[Tuple[int, int]], 
                                num_negative: int) -> List[Tuple[int, int]]:
        """Generate negative samples for training."""
        positive_set = set(positive_edges)
        negatives = []
        
        max_attempts = num_negative * 10  # Prevent infinite loops
        attempts = 0
        
        while len(negatives) < num_negative and attempts < max_attempts:
            source = np.random.randint(0, self.num_entities)
            target = np.random.randint(0, self.num_entities)
            
            if source != target and (source, target) not in positive_set:
                negatives.append((source, target))
                
            attempts += 1
            
        return negatives
    
    def create_training_data(self, 
                           test_size: float = 0.2, 
                           negative_ratio: int = 1,
                           random_state: int = 42) -> Dict[str, torch.Tensor]:
        """Create train/test splits with positive and negative samples."""
        if not self.citation_edges:
            raise ValueError("Must call extract_graph_data() first")
        
        # Split positive edges
        train_pos, test_pos = train_test_split(
            self.citation_edges, 
            test_size=test_size, 
            random_state=random_state
        )
        
        # Generate negative samples
        train_neg = self.generate_negative_samples(train_pos, len(train_pos) * negative_ratio)
        test_neg = self.generate_negative_samples(test_pos, len(test_pos) * negative_ratio)
        
        # Convert to tensors
        train_data = {
            "positive": torch.tensor(train_pos, dtype=torch.long),
            "negative": torch.tensor(train_neg, dtype=torch.long),
            "labels": torch.cat([
                torch.ones(len(train_pos)), 
                torch.zeros(len(train_neg))
            ])
        }
        
        test_data = {
            "positive": torch.tensor(test_pos, dtype=torch.long),
            "negative": torch.tensor(test_neg, dtype=torch.long), 
            "labels": torch.cat([
                torch.ones(len(test_pos)),
                torch.zeros(len(test_neg))
            ])
        }
        
        # Combine positive and negative for training
        train_edges = torch.cat([train_data["positive"], train_data["negative"]])
        test_edges = torch.cat([test_data["positive"], test_data["negative"]])
        
        return {
            "train_edges": train_edges,
            "train_labels": train_data["labels"],
            "test_edges": test_edges,
            "test_labels": test_data["labels"],
            "num_entities": self.num_entities,
            "entity_mapping": self.paper_to_id
        }
    
    def get_paper_metadata(self) -> pd.DataFrame:
        """Get detailed paper metadata for analysis."""
        return self.db.query(QUERIES["PAPER_METADATA"])
    
    def get_dataset_stats(self) -> Dict[str, any]:
        """Get comprehensive dataset statistics."""
        G = self.build_networkx_graph()
        
        stats = {
            "num_papers": self.num_entities,
            "num_citations": len(self.citation_edges),
            "density": nx.density(G),
            "avg_degree": sum(dict(G.degree()).values()) / self.num_entities,
            "is_connected": nx.is_weakly_connected(G),
            "num_components": nx.number_weakly_connected_components(G)
        }
        
        if nx.is_weakly_connected(G):
            stats["avg_path_length"] = nx.average_shortest_path_length(G.to_undirected())
            stats["clustering_coefficient"] = nx.average_clustering(G.to_undirected())
        
        return stats


def load_citation_graph(db: Neo4jConnection) -> CitationGraphExtractor:
    """Convenience function to load and extract citation graph data."""
    extractor = CitationGraphExtractor(db)
    extractor.extract_graph_data()
    return extractor
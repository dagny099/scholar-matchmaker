"""
Neo4j database connection and query utilities for citation network analysis.
Provides graph extraction capabilities for TransE model training.
"""

from __future__ import annotations
import os
from typing import Optional, Dict, Any, Tuple, List
import pandas as pd
import networkx as nx
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Cypher queries for graph extraction
QUERIES = {
    "CITATION_EDGES": """
        MATCH (source:Paper)-[:CITES]->(target:Paper)
        RETURN source.paperId as source_id, target.paperId as target_id
    """,
    
    "ALL_PAPERS": """
        MATCH (p:Paper)
        RETURN p.paperId as paper_id, p.title as title, 
               p.citationCount as citation_count, p.publicationDate as pub_date
    """,
    
    "PAPER_METADATA": """
        MATCH (p:Paper)
        OPTIONAL MATCH (p)-[:AUTHORED]-(a:Author)
        OPTIONAL MATCH (p)-[:PUBLISHED_IN]-(v:PubVenue)  
        OPTIONAL MATCH (p)-[:IS_ABOUT]-(f:Field)
        RETURN p.paperId as paper_id, p.title as title,
               collect(DISTINCT a.name) as authors,
               collect(DISTINCT v.name) as venues,
               collect(DISTINCT f.name) as fields,
               p.citationCount as citations
    """,
    
    "NETWORK_STATS": """
        MATCH (p:Paper)
        OPTIONAL MATCH (p)-[:CITES]->(cited:Paper)
        RETURN count(DISTINCT p) as num_papers,
               count(cited) as num_citations,
               count(DISTINCT cited) as num_cited_papers
    """
}


class Neo4jConnection:
    """Enhanced Neo4j connection with graph extraction capabilities."""
    
    def __init__(self):
        self.uri = os.getenv("NEO4J_URI")
        self.user = os.getenv("NEO4J_USER") 
        self.password = os.getenv("NEO4J_PWD") or os.getenv("NEO4J_PASSWORD")
        
        if not all([self.uri, self.user, self.password]):
            raise ValueError("Missing Neo4j credentials. Check .env file.")
            
        self._driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        
    def query(self, cypher: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Execute Cypher query and return results as DataFrame."""
        with self._driver.session() as session:
            result = session.run(cypher, params or {})
            return pd.DataFrame([r.data() for r in result])
    
    def test_connection(self) -> bool:
        """Test database connectivity."""
        try:
            self.query("RETURN 1 as test")
            return True
        except Exception:
            return False
    
    def get_network_stats(self) -> Dict[str, int]:
        """Get basic network statistics."""
        stats = self.query(QUERIES["NETWORK_STATS"]).iloc[0]
        return {
            "papers": int(stats["num_papers"]),
            "citations": int(stats["num_citations"]), 
            "cited_papers": int(stats["num_cited_papers"])
        }
    
    def close(self):
        """Close database connection."""
        if self._driver:
            self._driver.close()


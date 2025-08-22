"""
Visualization utilities for citation network analysis and TransE results.
Produces portfolio-quality matplotlib and seaborn plots.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import networkx as nx
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional, Any
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Set style for portfolio-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def plot_network_overview(G: nx.Graph, 
                         title: str = "Citation Network Overview",
                         figsize: Tuple[int, int] = (12, 8),
                         save_path: Optional[str] = None) -> plt.Figure:
    """
    Create an overview visualization of the citation network.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Left plot: Network graph
    pos = nx.spring_layout(G, k=1, iterations=50, seed=42)
    
    # Node sizes based on citation count
    node_sizes = []
    for node in G.nodes():
        citation_count = G.nodes[node].get('citation_count', 1)
        node_sizes.append(max(20, min(200, citation_count * 2)))
    
    # Draw network
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                          node_color='lightblue', alpha=0.7, ax=ax1)
    nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.3, 
                          arrows=True, arrowsize=10, ax=ax1)
    
    ax1.set_title("Citation Network Structure", fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Right plot: Degree distribution
    degrees = [d for n, d in G.degree()]
    ax2.hist(degrees, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.set_xlabel('Node Degree', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Degree Distribution', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.95)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_network_statistics(stats: Dict[str, Any],
                          figsize: Tuple[int, int] = (14, 6),
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Visualize comprehensive network statistics.
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()
    
    # Prepare data
    basic_stats = {
        'Papers': stats.get('num_papers', 0),
        'Citations': stats.get('num_citations', 0),
        'Avg Degree': stats.get('avg_degree', 0)
    }
    
    # Plot 1: Basic counts
    bars = axes[0].bar(basic_stats.keys(), basic_stats.values(), 
                      color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    axes[0].set_title('Network Scale', fontweight='bold')
    axes[0].set_ylabel('Count')
    
    # Add value labels on bars
    for bar, value in zip(bars, basic_stats.values()):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.0f}' if value > 1 else f'{value:.2f}',
                    ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Density and connectivity
    connectivity_data = {
        'Density': stats.get('density', 0),
        'Connected': 1.0 if stats.get('is_connected', False) else 0.0
    }
    
    bars = axes[1].bar(connectivity_data.keys(), connectivity_data.values(),
                      color=['#96CEB4', '#FFEAA7'])
    axes[1].set_title('Network Properties', fontweight='bold')
    axes[1].set_ylabel('Value')
    axes[1].set_ylim(0, 1)
    
    for bar, value in zip(bars, connectivity_data.values()):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: Path length and clustering (if available)
    if 'avg_path_length' in stats:
        structural_data = {
            'Avg Path Length': stats['avg_path_length'],
            'Clustering Coeff': stats.get('clustering_coefficient', 0)
        }
        bars = axes[2].bar(structural_data.keys(), structural_data.values(),
                          color=['#DDA0DD', '#98D8C8'])
        axes[2].set_title('Structural Properties', fontweight='bold')
        
        for bar, value in zip(bars, structural_data.values()):
            height = bar.get_height()
            axes[2].text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
    else:
        axes[2].text(0.5, 0.5, 'Network not\nfully connected', 
                    ha='center', va='center', transform=axes[2].transAxes,
                    fontsize=12, style='italic')
        axes[2].set_title('Structural Properties', fontweight='bold')
    
    # Plot 4: Components
    axes[3].bar(['Components'], [stats.get('num_components', 1)], 
               color='#FF7675')
    axes[3].set_title('Network Components', fontweight='bold')
    axes[3].set_ylabel('Count')
    
    # Plot 5: Empty for now
    axes[4].axis('off')
    
    # Plot 6: Summary text
    axes[5].axis('off')
    summary_text = f"""
    Network Summary:
    • {stats.get('num_papers', 0)} papers
    • {stats.get('num_citations', 0)} citations
    • Density: {stats.get('density', 0):.4f}
    • Connected: {'Yes' if stats.get('is_connected', False) else 'No'}
    • Components: {stats.get('num_components', 1)}
    """
    axes[5].text(0.05, 0.95, summary_text, transform=axes[5].transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.suptitle('Citation Network Statistics', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_training_history(loss_history: List[float],
                        figsize: Tuple[int, int] = (10, 6),
                        save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot TransE training loss over epochs.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    epochs = range(1, len(loss_history) + 1)
    ax.plot(epochs, loss_history, linewidth=2, color='#FF6B6B', marker='o', markersize=3)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Training Loss', fontsize=12)
    ax.set_title('TransE Model Training Progress', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add final loss value annotation
    final_loss = loss_history[-1]
    ax.annotate(f'Final Loss: {final_loss:.4f}', 
               xy=(len(loss_history), final_loss),
               xytext=(len(loss_history)*0.7, final_loss*1.1),
               arrowprops=dict(arrowstyle='->', color='black'),
               fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_evaluation_results(results: Dict[str, Any],
                          figsize: Tuple[int, int] = (12, 8),
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Visualize comprehensive evaluation results.
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Plot 1: Hits@K scores
    ax1 = fig.add_subplot(gs[0, 0])
    hits_data = results.get('hits', {})
    if hits_data:
        k_values = list(hits_data.keys())
        hit_scores = list(hits_data.values())
        
        bars = ax1.bar([f'Hits@{k}' for k in k_values], hit_scores, 
                      color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax1.set_title('Ranking Performance', fontweight='bold')
        ax1.set_ylabel('Hit Rate')
        ax1.set_ylim(0, 1)
        
        # Add value labels
        for bar, score in zip(bars, hit_scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: MRR Score
    ax2 = fig.add_subplot(gs[0, 1])
    mrr_score = results.get('mrr', 0)
    ax2.bar(['MRR'], [mrr_score], color='#96CEB4')
    ax2.set_title('Mean Reciprocal Rank', fontweight='bold')
    ax2.set_ylabel('MRR Score')
    ax2.set_ylim(0, 1)
    ax2.text(0, mrr_score, f'{mrr_score:.4f}', ha='center', va='bottom', 
            fontweight='bold', fontsize=12)
    
    # Plot 3: Binary Classification Metrics
    ax3 = fig.add_subplot(gs[0, 2])
    auc_score = results.get('auc', 0)
    ap_score = results.get('average_precision', 0)
    
    bars = ax3.bar(['AUC', 'AP'], [auc_score, ap_score], 
                  color=['#FFEAA7', '#DDA0DD'])
    ax3.set_title('Classification Performance', fontweight='bold')
    ax3.set_ylabel('Score')
    ax3.set_ylim(0, 1)
    
    for bar, score in zip(bars, [auc_score, ap_score]):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Performance summary radar chart
    ax4 = fig.add_subplot(gs[1, :])
    
    # Create summary text
    summary = f"""
    Model Evaluation Summary
    
    Ranking Metrics:
    • Mean Reciprocal Rank (MRR): {mrr_score:.4f}
    • Hits@1: {hits_data.get(1, 0):.3f} | Hits@3: {hits_data.get(3, 0):.3f} | Hits@10: {hits_data.get(10, 0):.3f}
    
    Classification Metrics:
    • AUC Score: {auc_score:.4f}
    • Average Precision: {ap_score:.4f}
    
    Interpretation:
    • MRR measures ranking quality (higher = better)
    • Hits@K shows proportion of correct predictions in top-K (higher = better)
    • AUC measures binary classification performance (higher = better)
    """
    
    ax4.text(0.05, 0.95, summary, transform=ax4.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.1))
    ax4.axis('off')
    
    plt.suptitle('TransE Model Evaluation Results', fontsize=16, fontweight='bold', y=0.95)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_embedding_visualization(embeddings: torch.Tensor,
                               paper_metadata: pd.DataFrame,
                               method: str = 'tsne',
                               figsize: Tuple[int, int] = (12, 8),
                               save_path: Optional[str] = None) -> plt.Figure:
    """
    Visualize paper embeddings in 2D using t-SNE or PCA.
    """
    # Convert to numpy
    emb_np = embeddings.detach().cpu().numpy()
    
    # Dimensionality reduction
    if method.lower() == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(emb_np)-1))
        coords = reducer.fit_transform(emb_np)
        title_suffix = "t-SNE"
    else:  # PCA
        reducer = PCA(n_components=2, random_state=42)
        coords = reducer.fit_transform(emb_np)
        title_suffix = "PCA"
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color by citation count if available
    if 'citations' in paper_metadata.columns:
        citations = paper_metadata['citations'].fillna(0)
        scatter = ax.scatter(coords[:, 0], coords[:, 1], 
                           c=citations, cmap='viridis', 
                           alpha=0.6, s=30)
        plt.colorbar(scatter, ax=ax, label='Citation Count')
    else:
        ax.scatter(coords[:, 0], coords[:, 1], alpha=0.6, s=30)
    
    ax.set_xlabel(f'{title_suffix} Component 1', fontsize=12)
    ax.set_ylabel(f'{title_suffix} Component 2', fontsize=12)
    ax.set_title(f'Paper Embedding Visualization ({title_suffix})', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_prediction_analysis(predictions_df: pd.DataFrame,
                           top_n: int = 20,
                           figsize: Tuple[int, int] = (14, 10),
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Analyze and visualize top citation predictions.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Plot 1: Score distribution
    axes[0,0].hist(predictions_df['score'], bins=30, alpha=0.7, 
                   color='skyblue', edgecolor='black')
    axes[0,0].set_xlabel('Prediction Score')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].set_title('Distribution of Prediction Scores', fontweight='bold')
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot 2: Top predictions by source paper
    top_preds = predictions_df.head(top_n)
    source_counts = top_preds['source_paper'].value_counts().head(10)
    
    axes[0,1].barh(range(len(source_counts)), source_counts.values,
                   color='lightcoral')
    axes[0,1].set_yticks(range(len(source_counts)))
    axes[0,1].set_yticklabels([s[:50] + '...' if len(s) > 50 else s 
                              for s in source_counts.index], fontsize=8)
    axes[0,1].set_xlabel('Number of Predictions')
    axes[0,1].set_title('Top Sources by Prediction Count', fontweight='bold')
    
    # Plot 3: Score vs. Rank
    sample_data = predictions_df.sample(min(1000, len(predictions_df)), 
                                       random_state=42)
    axes[1,0].scatter(sample_data['rank'], sample_data['score'], 
                      alpha=0.5, s=10)
    axes[1,0].set_xlabel('Rank')
    axes[1,0].set_ylabel('Score')
    axes[1,0].set_title('Score vs. Rank Relationship', fontweight='bold')
    axes[1,0].grid(True, alpha=0.3)
    
    # Plot 4: Top predictions table
    axes[1,1].axis('off')
    top_10 = predictions_df.head(10)[['source_paper', 'target_paper', 'score']]
    
    # Truncate long titles
    top_10_display = top_10.copy()
    top_10_display['source_paper'] = top_10_display['source_paper'].apply(
        lambda x: x[:30] + '...' if len(x) > 30 else x)
    top_10_display['target_paper'] = top_10_display['target_paper'].apply(
        lambda x: x[:30] + '...' if len(x) > 30 else x)
    top_10_display['score'] = top_10_display['score'].round(4)
    
    table = axes[1,1].table(cellText=top_10_display.values,
                           colLabels=['Source', 'Target', 'Score'],
                           cellLoc='left', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)
    axes[1,1].set_title('Top 10 Predictions', fontweight='bold', pad=20)
    
    plt.suptitle('Citation Prediction Analysis', fontsize=16, fontweight='bold', y=0.95)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


# Style configuration for consistent plots
def set_portfolio_style():
    """Set consistent styling for portfolio-quality plots."""
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16
    })


# Initialize style
set_portfolio_style()
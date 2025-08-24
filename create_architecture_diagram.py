"""
Professional Architecture Diagram Generator
Creates a comprehensive system architecture diagram for the Citation Network GNN project
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Arrow
import numpy as np

def create_architecture_diagram():
    """Create a professional system architecture diagram"""
    
    # Set up the figure with a professional color scheme
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Professional color scheme
    colors = {
        'data_layer': '#2C3E50',      # Dark blue-gray for data layer
        'processing': '#E74C3C',       # Red for processing/ML
        'analysis': '#3498DB',         # Blue for analysis
        'visualization': '#27AE60',    # Green for visualization
        'infrastructure': '#8E44AD',   # Purple for infrastructure
        'api': '#F39C12',             # Orange for APIs
        'text_color': '#2C3E50',       # Dark text
        'light_gray': '#ECF0F1',       # Light background
        'accent': '#E67E22'            # Orange accent
    }
    
    # Background
    background = Rectangle((0, 0), 10, 10, facecolor=colors['light_gray'], alpha=0.3, zorder=0)
    ax.add_patch(background)
    
    # Title
    ax.text(5, 9.6, 'Scholarly Matchmaking: Citation Prediction System Architecture', 
            ha='center', va='center', fontsize=18, fontweight='bold', color=colors['text_color'])
    ax.text(5, 9.3, 'Graph Neural Network Approach to Academic Citation Discovery', 
            ha='center', va='center', fontsize=12, style='italic', color=colors['text_color'])
    
    # Layer 1: Data Infrastructure Layer (Bottom)
    # Neo4j Database
    neo4j_box = FancyBboxPatch((0.5, 0.5), 2.5, 1.2, boxstyle="round,pad=0.1", 
                               facecolor=colors['data_layer'], edgecolor='white', linewidth=2)
    ax.add_patch(neo4j_box)
    ax.text(1.75, 1.4, 'Neo4j Aura Database', ha='center', va='center', 
            fontsize=10, fontweight='bold', color='white')
    ax.text(1.75, 1.1, '~12K Papers\n~19K Citations', ha='center', va='center', 
            fontsize=8, color='white')
    ax.text(1.75, 0.8, 'Graph Schema:\nPaper→Author→Venue→Field', ha='center', va='center', 
            fontsize=7, color='white')
    
    # Semantic Scholar API
    api_box = FancyBboxPatch((3.5, 0.5), 2, 1.2, boxstyle="round,pad=0.1", 
                             facecolor=colors['api'], edgecolor='white', linewidth=2)
    ax.add_patch(api_box)
    ax.text(4.5, 1.4, 'Semantic Scholar API', ha='center', va='center', 
            fontsize=10, fontweight='bold', color='white')
    ax.text(4.5, 1.1, 'Academic Data Source', ha='center', va='center', 
            fontsize=8, color='white')
    ax.text(4.5, 0.8, 'Metadata & Citations', ha='center', va='center', 
            fontsize=7, color='white')
    
    # Infrastructure
    infra_box = FancyBboxPatch((6, 0.5), 3, 1.2, boxstyle="round,pad=0.1", 
                               facecolor=colors['infrastructure'], edgecolor='white', linewidth=2)
    ax.add_patch(infra_box)
    ax.text(7.5, 1.4, 'Development Infrastructure', ha='center', va='center', 
            fontsize=10, fontweight='bold', color='white')
    ax.text(7.5, 1.1, 'Poetry • Jupyter • Git', ha='center', va='center', 
            fontsize=8, color='white')
    ax.text(7.5, 0.8, 'Type Checking • Testing • CI/CD', ha='center', va='center', 
            fontsize=7, color='white')
    
    # Layer 2: Data Processing Layer
    # Data Extraction
    extraction_box = FancyBboxPatch((0.5, 2.5), 2.2, 1.2, boxstyle="round,pad=0.1", 
                                   facecolor=colors['processing'], edgecolor='white', linewidth=2)
    ax.add_patch(extraction_box)
    ax.text(1.6, 3.4, 'Data Extraction Pipeline', ha='center', va='center', 
            fontsize=10, fontweight='bold', color='white')
    ax.text(1.6, 3.1, 'CitationGraphExtractor', ha='center', va='center', 
            fontsize=8, color='white')
    ax.text(1.6, 2.8, 'Neo4j → NetworkX → PyTorch', ha='center', va='center', 
            fontsize=7, color='white')
    
    # Graph Construction
    graph_box = FancyBboxPatch((3.1, 2.5), 2.2, 1.2, boxstyle="round,pad=0.1", 
                              facecolor=colors['processing'], edgecolor='white', linewidth=2)
    ax.add_patch(graph_box)
    ax.text(4.2, 3.4, 'Graph Construction', ha='center', va='center', 
            fontsize=10, fontweight='bold', color='white')
    ax.text(4.2, 3.1, 'NetworkX Integration', ha='center', va='center', 
            fontsize=8, color='white')
    ax.text(4.2, 2.8, 'Entity Mapping • Tensors', ha='center', va='center', 
            fontsize=7, color='white')
    
    # Preprocessing
    preprocess_box = FancyBboxPatch((5.7, 2.5), 2.2, 1.2, boxstyle="round,pad=0.1", 
                                   facecolor=colors['processing'], edgecolor='white', linewidth=2)
    ax.add_patch(preprocess_box)
    ax.text(6.8, 3.4, 'Data Preprocessing', ha='center', va='center', 
            fontsize=10, fontweight='bold', color='white')
    ax.text(6.8, 3.1, 'Train/Test Split', ha='center', va='center', 
            fontsize=8, color='white')
    ax.text(6.8, 2.8, 'Negative Sampling', ha='center', va='center', 
            fontsize=7, color='white')
    
    # Layer 3: Machine Learning Core
    # TransE Model
    model_box = FancyBboxPatch((1, 4.5), 3.5, 1.5, boxstyle="round,pad=0.1", 
                              facecolor=colors['analysis'], edgecolor='white', linewidth=2)
    ax.add_patch(model_box)
    ax.text(2.75, 5.6, 'TransE Graph Neural Network', ha='center', va='center', 
            fontsize=12, fontweight='bold', color='white')
    ax.text(2.75, 5.3, 'PyTorch Implementation', ha='center', va='center', 
            fontsize=9, color='white')
    ax.text(2.75, 5.0, 'Entity Embeddings (128D) • Relation Embeddings', ha='center', va='center', 
            fontsize=8, color='white')
    ax.text(2.75, 4.8, 'Margin Ranking Loss • Adam Optimizer', ha='center', va='center', 
            fontsize=7, color='white')
    
    # Training Pipeline
    training_box = FancyBboxPatch((5.5, 4.5), 2.5, 1.5, boxstyle="round,pad=0.1", 
                                 facecolor=colors['analysis'], edgecolor='white', linewidth=2)
    ax.add_patch(training_box)
    ax.text(6.75, 5.6, 'Training Pipeline', ha='center', va='center', 
            fontsize=11, fontweight='bold', color='white')
    ax.text(6.75, 5.3, 'Batch Processing', ha='center', va='center', 
            fontsize=9, color='white')
    ax.text(6.75, 5.0, 'Embedding Normalization', ha='center', va='center', 
            fontsize=8, color='white')
    ax.text(6.75, 4.8, 'Loss Tracking', ha='center', va='center', 
            fontsize=7, color='white')
    
    # Layer 4: Evaluation & Analysis
    # Evaluation Metrics
    eval_box = FancyBboxPatch((0.5, 6.5), 2.5, 1.2, boxstyle="round,pad=0.1", 
                             facecolor=colors['visualization'], edgecolor='white', linewidth=2)
    ax.add_patch(eval_box)
    ax.text(1.75, 7.4, 'Evaluation Framework', ha='center', va='center', 
            fontsize=10, fontweight='bold', color='white')
    ax.text(1.75, 7.1, 'MRR • Hits@K • AUC', ha='center', va='center', 
            fontsize=8, color='white')
    ax.text(1.75, 6.8, 'Binary Classification Metrics', ha='center', va='center', 
            fontsize=7, color='white')
    
    # Prediction Engine
    prediction_box = FancyBboxPatch((3.5, 6.5), 2.5, 1.2, boxstyle="round,pad=0.1", 
                                   facecolor=colors['visualization'], edgecolor='white', linewidth=2)
    ax.add_patch(prediction_box)
    ax.text(4.75, 7.4, 'Citation Prediction', ha='center', va='center', 
            fontsize=10, fontweight='bold', color='white')
    ax.text(4.75, 7.1, 'Link Prediction Engine', ha='center', va='center', 
            fontsize=8, color='white')
    ax.text(4.75, 6.8, 'Top-K Recommendations', ha='center', va='center', 
            fontsize=7, color='white')
    
    # Visualization Suite
    viz_box = FancyBboxPatch((6.5, 6.5), 2.5, 1.2, boxstyle="round,pad=0.1", 
                            facecolor=colors['visualization'], edgecolor='white', linewidth=2)
    ax.add_patch(viz_box)
    ax.text(7.75, 7.4, 'Visualization Suite', ha='center', va='center', 
            fontsize=10, fontweight='bold', color='white')
    ax.text(7.75, 7.1, 'Matplotlib • Seaborn', ha='center', va='center', 
            fontsize=8, color='white')
    ax.text(7.75, 6.8, 'Network • Embeddings • Metrics', ha='center', va='center', 
            fontsize=7, color='white')
    
    # Layer 5: Application Layer
    # Jupyter Notebooks
    notebook_box = FancyBboxPatch((1, 8.2), 7, 0.8, boxstyle="round,pad=0.1", 
                                 facecolor=colors['accent'], edgecolor='white', linewidth=2)
    ax.add_patch(notebook_box)
    ax.text(4.5, 8.6, 'Interactive Analysis Environment (Jupyter Notebooks)', ha='center', va='center', 
            fontsize=12, fontweight='bold', color='white')
    ax.text(4.5, 8.4, '01_exploration.ipynb • 02_training.ipynb • 03_evaluation.ipynb • 04_storytelling.ipynb', 
            ha='center', va='center', fontsize=9, color='white')
    
    # Add arrows to show data flow
    arrow_props = dict(arrowstyle='->', connectionstyle='arc3', color=colors['text_color'], lw=2)
    
    # Data layer to processing layer
    ax.annotate('', xy=(1.6, 2.5), xytext=(1.6, 1.7), arrowprops=arrow_props)
    ax.annotate('', xy=(4.2, 2.5), xytext=(4.5, 1.7), arrowprops=arrow_props)
    
    # Processing to ML
    ax.annotate('', xy=(2.75, 4.5), xytext=(3.2, 3.7), arrowprops=arrow_props)
    ax.annotate('', xy=(6.75, 4.5), xytext=(6.8, 3.7), arrowprops=arrow_props)
    
    # ML to Evaluation
    ax.annotate('', xy=(1.75, 6.5), xytext=(2.2, 6.0), arrowprops=arrow_props)
    ax.annotate('', xy=(4.75, 6.5), xytext=(4.2, 6.0), arrowprops=arrow_props)
    ax.annotate('', xy=(7.75, 6.5), xytext=(7.3, 6.0), arrowprops=arrow_props)
    
    # Evaluation to Notebooks
    ax.annotate('', xy=(4.5, 8.2), xytext=(4.5, 7.7), arrowprops=arrow_props)
    
    # Add technical specifications sidebar
    spec_box = FancyBboxPatch((8.3, 2.5), 1.5, 3.2, boxstyle="round,pad=0.1", 
                             facecolor='white', edgecolor=colors['text_color'], linewidth=1.5)
    ax.add_patch(spec_box)
    ax.text(9.05, 5.5, 'Tech Stack', ha='center', va='center', 
            fontsize=10, fontweight='bold', color=colors['text_color'])
    
    tech_specs = [
        'Python 3.10+',
        'PyTorch 2.0',
        'NetworkX 3.2',
        'Neo4j Driver',
        'Pandas 2.2',
        'Scikit-learn',
        'Matplotlib',
        'Seaborn',
        'Jupyter',
        'Poetry',
        'Type Hints',
        'Async I/O'
    ]
    
    for i, spec in enumerate(tech_specs):
        ax.text(9.05, 5.2 - i*0.15, f'• {spec}', ha='center', va='center', 
                fontsize=7, color=colors['text_color'])
    
    # Add legend
    legend_elements = [
        mpatches.Rectangle((0, 0), 1, 1, facecolor=colors['data_layer'], label='Data Layer'),
        mpatches.Rectangle((0, 0), 1, 1, facecolor=colors['processing'], label='Processing'),
        mpatches.Rectangle((0, 0), 1, 1, facecolor=colors['analysis'], label='ML Core'),
        mpatches.Rectangle((0, 0), 1, 1, facecolor=colors['visualization'], label='Analysis'),
        mpatches.Rectangle((0, 0), 1, 1, facecolor=colors['accent'], label='Interface'),
        mpatches.Rectangle((0, 0), 1, 1, facecolor=colors['api'], label='External API'),
        mpatches.Rectangle((0, 0), 1, 1, facecolor=colors['infrastructure'], label='Infrastructure')
    ]
    
    ax.legend(handles=legend_elements, loc='lower left', bbox_to_anchor=(0.02, 0.02), 
              frameon=True, fancybox=True, shadow=True, fontsize=8)
    
    # Add architectural principles text box
    principles_box = FancyBboxPatch((0.2, 8.7), 2.5, 1.1, boxstyle="round,pad=0.1", 
                                   facecolor='white', edgecolor=colors['text_color'], 
                                   linewidth=1.5, alpha=0.9)
    ax.add_patch(principles_box)
    ax.text(1.45, 9.5, 'Architecture Principles', ha='center', va='center', 
            fontsize=9, fontweight='bold', color=colors['text_color'])
    
    principles = [
        '• Modular Design',
        '• Type Safety',
        '• Scalable Processing',
        '• Reproducible Results',
        '• Portfolio Quality'
    ]
    
    for i, principle in enumerate(principles):
        ax.text(0.3, 9.3 - i*0.1, principle, ha='left', va='center', 
                fontsize=7, color=colors['text_color'])
    
    plt.tight_layout()
    return fig

def create_data_flow_diagram():
    """Create a detailed data flow diagram"""
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Colors
    colors = {
        'input': '#3498DB',
        'process': '#E74C3C',
        'output': '#27AE60',
        'storage': '#9B59B6',
        'text': '#2C3E50'
    }
    
    ax.text(5, 9.5, 'Data Flow & Processing Pipeline', 
            ha='center', va='center', fontsize=16, fontweight='bold', color=colors['text'])
    
    # Input Sources
    # Semantic Scholar
    ss_box = FancyBboxPatch((0.5, 7.5), 2, 1, boxstyle="round,pad=0.1", 
                           facecolor=colors['input'], edgecolor='white', linewidth=2)
    ax.add_patch(ss_box)
    ax.text(1.5, 8, 'Semantic Scholar\nAPI', ha='center', va='center', 
            fontsize=9, fontweight='bold', color='white')
    
    # Neo4j Database
    neo4j_box = FancyBboxPatch((0.5, 5.8), 2, 1, boxstyle="round,pad=0.1", 
                              facecolor=colors['storage'], edgecolor='white', linewidth=2)
    ax.add_patch(neo4j_box)
    ax.text(1.5, 6.3, 'Neo4j Aura\nDatabase', ha='center', va='center', 
            fontsize=9, fontweight='bold', color='white')
    
    # Data Extraction
    extract_box = FancyBboxPatch((3.5, 6.5), 2, 1.5, boxstyle="round,pad=0.1", 
                                facecolor=colors['process'], edgecolor='white', linewidth=2)
    ax.add_patch(extract_box)
    ax.text(4.5, 7.5, 'Data Extraction', ha='center', va='center', 
            fontsize=10, fontweight='bold', color='white')
    ax.text(4.5, 7.2, 'Cypher Queries', ha='center', va='center', 
            fontsize=8, color='white')
    ax.text(4.5, 6.9, 'Graph → DataFrame', ha='center', va='center', 
            fontsize=8, color='white')
    
    # Graph Construction
    graph_box = FancyBboxPatch((6.5, 6.5), 2.5, 1.5, boxstyle="round,pad=0.1", 
                              facecolor=colors['process'], edgecolor='white', linewidth=2)
    ax.add_patch(graph_box)
    ax.text(7.75, 7.5, 'Graph Construction', ha='center', va='center', 
            fontsize=10, fontweight='bold', color='white')
    ax.text(7.75, 7.2, 'NetworkX Graph', ha='center', va='center', 
            fontsize=8, color='white')
    ax.text(7.75, 6.9, 'Entity Mapping', ha='center', va='center', 
            fontsize=8, color='white')
    
    # Training Data Prep
    prep_box = FancyBboxPatch((2, 4.5), 3, 1.2, boxstyle="round,pad=0.1", 
                             facecolor=colors['process'], edgecolor='white', linewidth=2)
    ax.add_patch(prep_box)
    ax.text(3.5, 5.3, 'Training Data Preparation', ha='center', va='center', 
            fontsize=10, fontweight='bold', color='white')
    ax.text(3.5, 5.0, 'Positive/Negative Sampling', ha='center', va='center', 
            fontsize=8, color='white')
    ax.text(3.5, 4.8, 'Train/Test Split → PyTorch Tensors', ha='center', va='center', 
            fontsize=8, color='white')
    
    # Model Training
    train_box = FancyBboxPatch((6, 4.5), 2.5, 1.2, boxstyle="round,pad=0.1", 
                              facecolor=colors['process'], edgecolor='white', linewidth=2)
    ax.add_patch(train_box)
    ax.text(7.25, 5.3, 'TransE Training', ha='center', va='center', 
            fontsize=10, fontweight='bold', color='white')
    ax.text(7.25, 5.0, 'Embedding Learning', ha='center', va='center', 
            fontsize=8, color='white')
    ax.text(7.25, 4.8, 'Margin Ranking Loss', ha='center', va='center', 
            fontsize=8, color='white')
    
    # Outputs
    # Trained Model
    model_box = FancyBboxPatch((1, 2.5), 2, 1, boxstyle="round,pad=0.1", 
                              facecolor=colors['output'], edgecolor='white', linewidth=2)
    ax.add_patch(model_box)
    ax.text(2, 3, 'Trained Model\nEmbeddings', ha='center', va='center', 
            fontsize=9, fontweight='bold', color='white')
    
    # Predictions
    pred_box = FancyBboxPatch((4, 2.5), 2, 1, boxstyle="round,pad=0.1", 
                             facecolor=colors['output'], edgecolor='white', linewidth=2)
    ax.add_patch(pred_box)
    ax.text(5, 3, 'Citation\nPredictions', ha='center', va='center', 
            fontsize=9, fontweight='bold', color='white')
    
    # Visualizations
    viz_box = FancyBboxPatch((7, 2.5), 2, 1, boxstyle="round,pad=0.1", 
                            facecolor=colors['output'], edgecolor='white', linewidth=2)
    ax.add_patch(viz_box)
    ax.text(8, 3, 'Analysis\nVisualizations', ha='center', va='center', 
            fontsize=9, fontweight='bold', color='white')
    
    # Add flow arrows
    arrow_props = dict(arrowstyle='->', connectionstyle='arc3', color='#34495E', lw=2)
    
    # API to extraction
    ax.annotate('', xy=(3.5, 7.2), xytext=(2.5, 8), arrowprops=arrow_props)
    # DB to extraction
    ax.annotate('', xy=(3.5, 6.8), xytext=(2.5, 6.3), arrowprops=arrow_props)
    # Extraction to graph construction
    ax.annotate('', xy=(6.5, 7.2), xytext=(5.5, 7.2), arrowprops=arrow_props)
    # Graph to training prep
    ax.annotate('', xy=(3.5, 5.7), xytext=(7, 6.5), arrowprops=arrow_props)
    # Prep to training
    ax.annotate('', xy=(6, 5.1), xytext=(5, 5.1), arrowprops=arrow_props)
    # Training to outputs
    ax.annotate('', xy=(2, 3.5), xytext=(6.5, 4.5), arrowprops=arrow_props)
    ax.annotate('', xy=(5, 3.5), xytext=(7, 4.5), arrowprops=arrow_props)
    ax.annotate('', xy=(8, 3.5), xytext=(7.5, 4.5), arrowprops=arrow_props)
    
    # Add data volume annotations
    ax.text(2.8, 7.8, '~12K papers\n~19K citations', ha='center', va='center', 
            fontsize=7, style='italic', color=colors['text'])
    ax.text(5.8, 7.8, 'Entity mappings\nGraph topology', ha='center', va='center', 
            fontsize=7, style='italic', color=colors['text'])
    ax.text(3.5, 4.2, '80/20 split\n~15K train edges', ha='center', va='center', 
            fontsize=7, style='italic', color=colors['text'])
    ax.text(7.25, 4.2, '128D embeddings\n100 epochs', ha='center', va='center', 
            fontsize=7, style='italic', color=colors['text'])
    
    # Add performance metrics box
    perf_box = FancyBboxPatch((1, 0.5), 8, 1.2, boxstyle="round,pad=0.1", 
                             facecolor='#ECF0F1', edgecolor=colors['text'], linewidth=1)
    ax.add_patch(perf_box)
    ax.text(5, 1.4, 'Key Performance Metrics & Capabilities', ha='center', va='center', 
            fontsize=11, fontweight='bold', color=colors['text'])
    
    metrics_text = """
• Network Scale: 12,553 papers with 18,912 citation relationships (density: 0.00012)
• Model Architecture: 128-dimensional TransE embeddings with margin ranking loss
• Evaluation: Mean Reciprocal Rank (MRR), Hits@K, AUC for comprehensive assessment
• Prediction Capability: Identify high-confidence missing citations from 157M+ potential links
• Visualization: Portfolio-quality network analysis, degree distributions, embedding projections
"""
    
    ax.text(5, 0.9, metrics_text, ha='center', va='center', 
            fontsize=8, color=colors['text'])
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    # Create architecture diagram
    print("Creating system architecture diagram...")
    arch_fig = create_architecture_diagram()
    arch_fig.savefig('/Users/bhs/PROJECTS/citation-map-dashboard/outputs/system_architecture.png', 
                     dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    
    # Create data flow diagram  
    print("Creating data flow diagram...")
    flow_fig = create_data_flow_diagram()
    flow_fig.savefig('/Users/bhs/PROJECTS/citation-map-dashboard/outputs/data_flow_architecture.png', 
                     dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    
    print("Architecture diagrams saved to outputs/ directory")
    print("✅ Professional architecture diagrams created successfully!")
    
    # plt.show()  # Disabled for non-interactive execution
    plt.close('all')  # Close figures to free memory
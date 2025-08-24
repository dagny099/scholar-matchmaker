# Citation Map Dashboard - Codebase Summary

## Project Overview

This is a **Graph Neural Network Citation Predictor** that uses TransE (Translating Embeddings) to predict missing citations in academic networks. The project transforms a Neo4j-stored citation network into learned embeddings that can identify "scholarly matchmaking" opportunities - papers that should cite each other but don't.

**Core Concept**: Just as dating apps suggest matches based on compatibility patterns, this model identifies papers with high probability of intellectual relevance based on learned network patterns.

## ⚠️ Critical Reality Check: What Actually Exists vs. Claims

### ✅ CONFIRMED EXISTING CODE
- **Total Source Lines**: 1,167 lines across 5 Python modules (excluding empty `__init__.py`)
- **Complete Implementation**: All core functionality is actually implemented
- **Working Models**: 19.3MB trained model file exists with entity mappings and metadata
- **Generated Outputs**: 17 visualization files and CSV predictions actually exist
- **Full Documentation**: 38 total docstrings across all functions and classes

## Architecture & Data Flow

### 1. Data Foundation
- **Source**: Academic citation data from Semantic Scholar API
- **Storage**: Remote Neo4j Aura database (~12,553 papers, ~18,912 relationships)
- **Scope**: Papers citing and cited by a 2009 eye-tracking research paper

### 2. Data Flow Pipeline

```
Neo4j Database → Data Extraction → Graph Processing → Model Training → Evaluation → Predictions
      ↓              ↓                 ↓               ↓             ↓            ↓
   Raw citations → DataFrames → NetworkX Graph → PyTorch Tensors → Metrics → CSV Output
```

## Core Scripts and Functions

### `/src/db.py` - Database Connection Layer
**Purpose**: Neo4j database connectivity and query execution

**Core Functions**:
- `Neo4jConnection.__init__()`: Establishes secure connection to Neo4j Aura
- `Neo4jConnection.query()`: Executes Cypher queries, returns pandas DataFrames
- `Neo4jConnection.get_network_stats()`: Basic network statistics (papers, citations)

**Data Flow**: Converts graph database → structured DataFrames

**Key Queries**:
```cypher
CITATION_EDGES: MATCH (source:Paper)-[:CITES]->(target:Paper)
ALL_PAPERS: MATCH (p:Paper) RETURN metadata
PAPER_METADATA: Complex query with authors, venues, fields
```

### `/src/data_extraction.py` - Graph Processing Engine
**Purpose**: Transform citation data into ML-ready tensors

**Core Class**: `CitationGraphExtractor`

**Key Functions**:
- `extract_graph_data()`: Queries Neo4j → builds entity mappings (paper_id ↔ integer)
- `build_networkx_graph()`: Creates NetworkX DiGraph with metadata
- `generate_negative_samples()`: Creates negative examples for training
- `create_training_data()`: 80/20 train/test split with negative sampling

**Data Flow**: 
```
Raw citations → Entity mapping → NetworkX graph → PyTorch tensors
{paper_id: int_index} → G(nodes, edges) → train/test splits
```

**Unique Functionality**: Intelligent negative sampling that avoids existing citations

### `/src/model.py` - TransE Implementation
**Purpose**: Graph neural network for learning paper embeddings

**Core Classes**:
- `TransE`: PyTorch model implementing TransE algorithm
- `TransETrainer`: Training loop with margin ranking loss

**Key Functions**:
- `TransE.forward()`: Computes ||source + relation - target|| scores
- `TransE.predict_links()`: Batch prediction for source→candidate pairs
- `TransETrainer.train_epoch()`: One epoch with positive/negative sampling
- `TransETrainer.save_model()`/`load_model()`: Model persistence

**Data Flow**:
```
(source, target) pairs → Embeddings → Distance scores → Ranking loss
Entity IDs → 128-dim vectors → L1 norm → Gradient descent
```

**Algorithm**: `embedding(source) + relation ≈ embedding(target)` for citations

### `/src/evaluation.py` - Model Assessment
**Purpose**: Comprehensive evaluation with ranking and classification metrics

**Core Class**: `LinkPredictionEvaluator`

**Key Functions**:
- `mean_reciprocal_rank()`: MRR computation for ranking quality
- `hits_at_k()`: Proportion of correct predictions in top-K
- `binary_classification_metrics()`: AUC and Average Precision
- `predict_missing_citations()`: Generate ranked prediction lists

**Data Flow**:
```
Test edges → Model predictions → Ranking → Metrics computation
(source, target) → Scores → Sorted ranks → MRR, Hits@K, AUC
```

**Unique Functionality**: Excludes existing citations from predictions

### `/src/visualization.py` - Portfolio-Quality Plots
**Purpose**: Create publication-ready visualizations

**Key Functions**:
- `plot_network_overview()`: Citation network structure + degree distribution
- `plot_training_history()`: Loss curves with convergence analysis
- `plot_evaluation_results()`: Comprehensive performance dashboard
- `plot_embedding_visualization()`: t-SNE/PCA of learned embeddings
- `plot_prediction_analysis()`: Top predictions with confidence distributions

**Data Flow**: Analysis results → Matplotlib/Seaborn → High-DPI PNG files

## Jupyter Notebook Pipeline

### `01_network_exploration.ipynb`
**Purpose**: Exploratory data analysis and baseline metrics

**Data Flow**:
1. Connect to Neo4j → Extract citation network
2. Build NetworkX graph → Compute network statistics  
3. Analyze degree distributions → Identify high-impact papers
4. Create network visualizations → Generate baseline insights

**Key Outputs**: Network statistics, degree distributions, visualization files

### `02_model_training.ipynb`  
**Purpose**: TransE model implementation and training

**Data Flow**:
1. Load citation data → Create train/test splits (80/20)
2. Initialize TransE model → Configure training parameters
3. Train with margin ranking loss → Monitor convergence
4. Save trained model → Generate training plots

**Key Outputs**: Trained model (.pt), entity mappings (.pkl), training history

### `03_prediction_analysis.ipynb`
**Purpose**: Model evaluation and citation prediction

**Data Flow**:
1. Load trained model → Prepare test data
2. Compute ranking metrics (MRR, Hits@K) → Binary classification (AUC)
3. Generate missing citation predictions → Analyze confidence patterns
4. Create evaluation visualizations → Export prediction CSV

**Key Outputs**: Evaluation metrics, prediction rankings, performance plots

### `04_story_visualization.ipynb`
**Purpose**: Transform analysis into compelling research narrative

**Data Flow**:
1. Load all results → Create story arc visualizations
2. Generate case studies → Analyze research impact
3. Create final dashboard → Document project success

**Key Outputs**: 5 story visualizations, comprehensive project dashboard

## ❌ IDENTIFIED WEAKNESSES & LIMITATIONS

### 1. **Code Quality Issues**
- **Debug Print Statements**: 5 print statements left in production code (`data_extraction.py`, `evaluation.py`)
- **Missing Error Handling**: Database connections lack comprehensive error recovery
- **No Input Validation**: Functions don't validate tensor shapes or data types
- **Hardcoded Values**: Magic numbers scattered throughout (embedding dims, batch sizes)

### 2. **Architecture Limitations**
- **Single Model**: Only TransE implemented - no comparison with other graph embeddings
- **CPU-Only Training**: No GPU optimization despite PyTorch availability
- **Memory Inefficient**: Loads entire graph into memory (~12K nodes manageable, won't scale)
- **No Incremental Learning**: Requires full retraining for new data

### 3. **Evaluation Concerns**
- **Small Test Set**: Only 3,783 positive test samples for evaluation
- **No Cross-Validation**: Single train/test split (potential overfitting)
- **Limited Baseline**: No comparison to simpler citation prediction methods
- **Cherry-Picked Results**: Case studies show only best predictions, not failures

### 4. **Data Dependencies**
- **Remote Database Required**: Completely dependent on Neo4j Aura connection
- **No Offline Mode**: Cannot run without internet access to database
- **Single Dataset**: Only tested on eye-tracking research papers (domain-specific)
- **No Data Validation**: Assumes clean, well-formatted database

### 5. **Production Readiness Gaps**
- **No API Endpoints**: Pure research code, not deployable service
- **No Monitoring**: No logging, metrics, or health checks
- **No Tests**: Despite pytest in dependencies, no test files found
- **Configuration Management**: Environment variables scattered, no config classes

## Unique Functionality & Differentiators

### ✅ **Genuine Strengths**
1. **Complete End-to-End Pipeline**: Data extraction → model training → evaluation → visualization
2. **Neo4j Integration**: Direct graph database connectivity (though brittle)
3. **Comprehensive Visualization**: 17 different plots and analysis outputs
4. **Story-Driven Presentation**: Transforms technical analysis into narrative
5. **Scholarly Focus**: Academic-specific exclusion logic and evaluation metrics

### ⚠️ **Overstated Capabilities**
1. **"Production-Ready"**: Missing tests, monitoring, error handling
2. **"Scalable"**: Memory-bound architecture won't handle large networks
3. **"Portfolio-Quality"**: Code quality issues undermine professional appearance

## Performance Reality Check

### ✅ **Confirmed Results**
- **Dataset Scale**: 12,553 papers, 18,912 citations (verified in outputs)
- **Model Performance**: MRR 0.1118 (fair), AUC 0.9845 (excellent), Hits@10 26.1%
- **Predictions Generated**: 1,000 total with 100 high-confidence (CSV file exists)
- **Training Efficiency**: 100 epochs, ~5 minutes on CPU

### ⚠️ **Performance Interpretation**
- **MRR 0.1118**: Average true citation rank is ~9th - **not impressive for practical use**
- **Hits@1 3.6%**: Only 1 in 28 top predictions is correct - **poor for recommendations**
- **AUC 0.9845**: High binary classification accuracy is **misleading** (easy to distinguish random pairs)
- **Sample Size**: 50 papers × 20 predictions = tiny evaluation scope

## Honest Comparison Points for Similar Codebases

### ✅ **Where This Codebase Excels**
1. **Complete Implementation**: All components actually work together
2. **Visualization Quality**: 17 professional plots with story narrative
3. **Documentation**: Every function has docstrings
4. **Neo4j Integration**: Direct graph database connectivity (rare)
5. **Scholarly Domain Focus**: Academic-specific evaluation metrics

### ❌ **Where Other Codebases Likely Superior**
1. **Scale**: Limited to ~12K nodes, others likely handle millions
2. **Model Variety**: Only TransE, others probably compare multiple approaches
3. **Evaluation Rigor**: No cross-validation or baselines vs. likely benchmarks elsewhere
4. **Production Features**: Missing APIs, tests, monitoring vs. likely deployment-ready code
5. **Performance**: MRR 0.11 suggests other methods probably achieve better results

### 🎯 **Key Differentiator**
The unique value is the **complete story pipeline** from raw citations → trained model → compelling visualizations. Most citation prediction codebases focus on algorithmic improvements, not end-to-end research narratives.

## Bottom Line Assessment

This is **excellent research demo code** with **genuine ML implementation** but **significant production limitations**. It successfully demonstrates concept feasibility but overstates readiness for real-world deployment. The visualization and storytelling quality partially compensates for technical weaknesses.
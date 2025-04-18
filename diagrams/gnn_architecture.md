# GNN Model Architecture for Fraud Pattern Recognition

## Overview

This document outlines the Graph Neural Network (GNN) architecture for pattern recognition in the credit card fraud detection system. The GNN model is designed to learn from the graph structure of transaction data to identify complex fraud patterns that may not be detectable through traditional rule-based or statistical methods.

## GNN Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                 Input Graph Structure                                │
│                                                                                     │
│    ┌─────┐        ┌─────┐        ┌─────┐        ┌─────┐        ┌─────┐              │
│    │User │────────│Card │────────│Trans│────────│Merch│────────│Locat│              │
│    └─────┘        └─────┘        └─────┘        └─────┘        └─────┘              │
│       │              │              │              │              │                 │
│       │              │              │              │              │                 │
│    ┌─────┐        ┌─────┐        ┌─────┐                                           │
│    │Device│────────│ IP  │────────│Alert│                                           │
│    └─────┘        └─────┘        └─────┘                                           │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                 Feature Extraction                                   │
│                                                                                     │
│    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐                │
│    │ Node Features   │    │ Edge Features   │    │ Graph Features  │                │
│    └─────────────────┘    └─────────────────┘    └─────────────────┘                │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                 GNN Model Layers                                     │
│                                                                                     │
│    ┌─────────────────────────────────────────────────────────────────────────────┐  │
│    │                          Message Passing Layers                             │  │
│    │                                                                             │  │
│    │    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐       │  │
│    │    │ Graph Attention │    │ GraphSAGE       │    │ Graph Isomorphic│       │  │
│    │    │ Layer (GAT)     │    │ Convolution     │    │ Network (GIN)   │       │  │
│    │    └─────────────────┘    └─────────────────┘    └─────────────────┘       │  │
│    └─────────────────────────────────────────────────────────────────────────────┘  │
│                                          │                                          │
│                                          ▼                                          │
│    ┌─────────────────────────────────────────────────────────────────────────────┐  │
│    │                          Readout/Pooling Layers                             │  │
│    │                                                                             │  │
│    │    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐       │  │
│    │    │ Global Mean     │    │ Global Max      │    │ Hierarchical    │       │  │
│    │    │ Pooling         │    │ Pooling         │    │ Pooling         │       │  │
│    │    └─────────────────┘    └─────────────────┘    └─────────────────┘       │  │
│    └─────────────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                 Prediction Layers                                    │
│                                                                                     │
│    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐                │
│    │ Fully Connected │    │ Dropout         │    │ Output Layer    │                │
│    │ Layers          │    │                 │    │ (Sigmoid)       │                │
│    └─────────────────┘    └─────────────────┘    └─────────────────┘                │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                 Model Outputs                                        │
│                                                                                     │
│    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐                │
│    │ Transaction     │    │ Pattern         │    │ Explanation     │                │
│    │ Fraud Score     │    │ Identification  │    │ Components      │                │
│    └─────────────────┘    └─────────────────┘    └─────────────────┘                │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

## Input Graph Structure

The GNN model operates on a heterogeneous graph structure derived from the Neo4j database, with multiple node and edge types:

### Node Types
- User nodes
- Card nodes
- Transaction nodes
- Merchant nodes
- Location nodes
- Device nodes
- IP Address nodes
- Alert nodes

### Edge Types
- User-Card edges (OWNS)
- Card-Transaction edges (MADE)
- Transaction-Merchant edges (AT)
- Transaction-Location edges (IN)
- Transaction-Device edges (USING)
- Device-IP edges (FROM)
- Transaction-Transaction edges (FOLLOWED_BY)
- Transaction-Alert edges (TRIGGERED)

## Feature Extraction

### Node Features

#### User Node Features
- Account age (normalized)
- Risk score
- Transaction frequency
- Average transaction amount
- Number of cards
- Fraud history flag
- KYC verification status
- One-hot encoded user segments

#### Card Node Features
- Card age (normalized)
- Card type (one-hot encoded)
- Credit limit (normalized)
- Utilization ratio
- Is active flag
- Is blocked flag

#### Transaction Node Features
- Amount (normalized)
- Hour of day (sine-cosine encoded)
- Day of week (sine-cosine encoded)
- Is online flag
- Transaction type (one-hot encoded)
- Payment method (one-hot encoded)
- MCC code (embedded)
- Response code (one-hot encoded)

#### Merchant Node Features
- Risk score
- Fraud rate
- Average transaction amount (normalized)
- Category (embedded)
- Is high risk flag

#### Location Node Features
- Country (embedded)
- Risk score
- Latitude and longitude (normalized)

#### Device Node Features
- Device type (one-hot encoded)
- Is mobile flag
- Is known device flag
- Risk score

#### IP Address Node Features
- Country (embedded)
- Is proxy flag
- Is VPN flag
- Is Tor flag
- Risk score

### Edge Features

#### OWNS Edge Features
- Relationship duration (normalized)

#### MADE Edge Features
- Time since card creation (normalized)

#### FOLLOWED_BY Edge Features
- Time difference (normalized)
- Distance difference (normalized)
- Velocity anomaly flag

#### AT Edge Features
- Is frequent merchant flag
- Visit count (normalized)

#### IN Edge Features
- Is typical location flag

#### USING Edge Features
- Is first use flag
- Use count (normalized)

### Graph-Level Features
- Number of transactions in time window
- Transaction velocity
- Geographic spread
- Merchant diversity
- Average transaction amount

## GNN Model Architecture

### Message Passing Layers

The GNN model uses a combination of message passing layers to learn node representations:

#### 1. Graph Attention Network (GAT) Layer
- Attention mechanism to weigh neighbor importance
- Multi-head attention for robust feature learning
- Handles heterogeneous node and edge types

```python
class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=8, dropout=0.1, edge_dim=None):
        super(GATLayer, self).__init__()
        self.gat = GATConv(
            in_channels=in_dim,
            out_channels=out_dim // num_heads,
            heads=num_heads,
            dropout=dropout,
            edge_dim=edge_dim
        )
        
    def forward(self, x, edge_index, edge_attr=None):
        return self.gat(x, edge_index, edge_attr)
```

#### 2. GraphSAGE Convolution Layer
- Aggregates information from local neighborhood
- Maintains node-specific information
- Scales well to large graphs

```python
class GraphSAGELayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GraphSAGELayer, self).__init__()
        self.sage = SAGEConv(in_dim, out_dim)
        
    def forward(self, x, edge_index):
        return self.sage(x, edge_index)
```

#### 3. Graph Isomorphic Network (GIN) Layer
- Powerful discriminative capabilities
- Captures complex structural patterns
- Theoretically most expressive GNN

```python
class GINLayer(nn.Module):
    def __init__(self, in_dim, out_dim, eps=0):
        super(GINLayer, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
        self.gin = GINConv(self.mlp, eps=eps)
        
    def forward(self, x, edge_index):
        return self.gin(x, edge_index)
```

### Heterogeneous Graph Neural Network

To handle the heterogeneous nature of the transaction graph, we use a Heterogeneous GNN architecture:

```python
class HeteroGNN(nn.Module):
    def __init__(self, metadata, hidden_channels, num_layers):
        super(HeteroGNN, self).__init__()
        
        # Initialize node embeddings for each node type
        self.node_embeddings = nn.ModuleDict()
        for node_type, feature_dim in node_feature_dims.items():
            self.node_embeddings[node_type] = nn.Linear(feature_dim, hidden_channels)
        
        # Initialize heterogeneous GNN layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('user', 'owns', 'card'): GATConv(hidden_channels, hidden_channels),
                ('card', 'made', 'transaction'): GATConv(hidden_channels, hidden_channels),
                ('transaction', 'at', 'merchant'): GATConv(hidden_channels, hidden_channels),
                ('transaction', 'in', 'location'): GATConv(hidden_channels, hidden_channels),
                ('transaction', 'using', 'device'): GATConv(hidden_channels, hidden_channels),
                ('device', 'from', 'ip_address'): GATConv(hidden_channels, hidden_channels),
                ('transaction', 'followed_by', 'transaction'): GATConv(hidden_channels, hidden_channels),
                ('transaction', 'triggered', 'alert'): GATConv(hidden_channels, hidden_channels),
                # Reverse edges
                ('card', 'owned_by', 'user'): GATConv(hidden_channels, hidden_channels),
                ('transaction', 'made_with', 'card'): GATConv(hidden_channels, hidden_channels),
                # ... other reverse edges
            })
            self.convs.append(conv)
        
        # Output layer for transaction nodes
        self.transaction_classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x_dict, edge_indices_dict):
        # Initial node embeddings
        for node_type, x in x_dict.items():
            x_dict[node_type] = self.node_embeddings[node_type](x)
        
        # Message passing layers
        for conv in self.convs:
            x_dict = conv(x_dict, edge_indices_dict)
            # Apply ReLU and dropout
            for node_type in x_dict.keys():
                x_dict[node_type] = F.relu(x_dict[node_type])
                x_dict[node_type] = F.dropout(x_dict[node_type], p=0.2, training=self.training)
        
        # Predict fraud probability for transaction nodes
        fraud_scores = self.transaction_classifier(x_dict['transaction'])
        
        return fraud_scores, x_dict
```

### Readout/Pooling Layers

For tasks that require graph-level predictions (e.g., identifying fraud patterns across multiple transactions), we use pooling layers:

#### 1. Global Mean Pooling
- Averages node features across the graph
- Provides a general overview of the graph

#### 2. Global Max Pooling
- Captures the most prominent features
- Useful for detecting strong fraud signals

#### 3. Hierarchical Pooling
- Progressively coarsens the graph
- Preserves structural information at different scales

```python
class GraphPooling(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(GraphPooling, self).__init__()
        self.mean_pool = global_mean_pool
        self.max_pool = global_max_pool
        self.hierarchical_pool = SAGPooling(in_dim, ratio=0.5)
        self.combine = nn.Linear(in_dim * 3, hidden_dim)
        
    def forward(self, x, edge_index, batch):
        # Mean pooling
        mean_x = self.mean_pool(x, batch)
        
        # Max pooling
        max_x = self.max_pool(x, batch)
        
        # Hierarchical pooling
        hier_x, _, _, batch_hier, _, _ = self.hierarchical_pool(x, edge_index, batch=batch)
        hier_x = self.mean_pool(hier_x, batch_hier)
        
        # Combine pooled representations
        combined = torch.cat([mean_x, max_x, hier_x], dim=1)
        return self.combine(combined)
```

### Prediction Layers

The final prediction layers convert node and graph embeddings into fraud predictions:

#### 1. Fully Connected Layers
- Transform embeddings into prediction-ready features
- Multiple layers with non-linearities for complex patterns

#### 2. Dropout
- Prevents overfitting
- Improves generalization to unseen transactions

#### 3. Output Layer
- Sigmoid activation for fraud probability
- Threshold-based decision making

```python
class PredictionHead(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(PredictionHead, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.fc(x)
```

## Model Outputs

### 1. Transaction Fraud Score
- Probability of fraud for each transaction
- Range: [0, 1]
- Threshold-based classification

### 2. Pattern Identification
- Identification of known fraud patterns
- Similarity scores to known patterns
- Discovery of new patterns

### 3. Explanation Components
- Feature importance scores
- Subgraph highlighting
- Critical path identification

## Training Approach

### 1. Loss Function
- Binary Cross Entropy for fraud classification
- Focal Loss to handle class imbalance
- Graph Contrastive Loss for pattern learning

```python
def loss_function(predictions, labels, pos_weight=10.0):
    # Binary cross entropy with logits
    bce = F.binary_cross_entropy(predictions, labels)
    
    # Focal loss component to handle class imbalance
    gamma = 2.0
    pt = predictions * labels + (1 - predictions) * (1 - labels)
    focal_weight = (1 - pt) ** gamma
    focal_loss = F.binary_cross_entropy(predictions, labels, weight=focal_weight)
    
    # Combine losses
    combined_loss = 0.5 * bce + 0.5 * focal_loss
    
    return combined_loss
```

### 2. Optimization
- Adam optimizer with learning rate scheduling
- Gradient clipping to prevent exploding gradients
- Early stopping based on validation performance

### 3. Sampling Strategy
- Balanced mini-batch sampling
- Negative sampling for large graphs
- Time-based sampling for temporal patterns

## Temporal Aspects

### 1. Temporal Encoding
- Time encodings for transaction timestamps
- Relative time differences between connected transactions
- Sequence modeling for transaction chains

### 2. Temporal Graph Construction
- Sliding window approach for graph construction
- Dynamic graph updates for real-time processing
- Historical context incorporation

### 3. Temporal Attention
- Attention mechanisms that consider temporal proximity
- Decay functions for older transactions
- Emphasis on recent behavioral changes

## Explainability Features

### 1. Attention Visualization
- Visualize attention weights between nodes
- Highlight important connections for fraud detection
- User-friendly interface for investigation

### 2. Feature Attribution
- SHAP values for feature importance
- Counterfactual explanations
- Critical subgraph identification

### 3. Pattern Explanation
- Mapping to known fraud patterns
- Similarity metrics to previous cases
- Natural language explanations

## Integration with Multi-Agent System

### 1. Agent Interaction
- GNN model serves as a core component for the Pattern Detection Agent
- Provides embeddings and predictions to other agents
- Receives feedback from Decision and Learning Agents

### 2. Model Context Protocol (MCP) Integration
- GNN outputs are formatted according to MCP
- Multi-modal context integration with transaction data
- Context sharing across agent boundaries

### 3. Neo4j Integration via ADK
- Graph data extraction from Neo4j for GNN training
- Real-time graph queries for model inference
- Prediction results storage in Neo4j

## Performance Optimization

### 1. Model Quantization
- Reduced precision for faster inference
- Minimal accuracy impact
- Deployment-friendly model size

### 2. Batched Processing
- Efficient batching of transaction subgraphs
- Parallel processing of independent components
- GPU acceleration for large-scale processing

### 3. Incremental Learning
- Continuous model updating with new data
- Transfer learning from previous models
- Adaptation to evolving fraud patterns

## Evaluation Metrics

### 1. Classification Metrics
- Precision, Recall, F1-Score
- Area Under ROC Curve (AUC-ROC)
- Area Under Precision-Recall Curve (AUC-PR)

### 2. Graph-Specific Metrics
- Link prediction accuracy
- Subgraph detection precision
- Pattern matching recall

### 3. Operational Metrics
- Inference time
- Memory usage
- Scalability with graph size

## Implementation Technologies

- **PyTorch Geometric**: Core GNN implementation
- **DGL (Deep Graph Library)**: Alternative for heterogeneous graphs
- **Neo4j Graph Data Science Library**: Graph algorithms and feature extraction
- **RAPIDS cuGraph**: GPU acceleration for large graphs
- **PyTorch Lightning**: Training workflow management
- **Weights & Biases**: Experiment tracking and visualization

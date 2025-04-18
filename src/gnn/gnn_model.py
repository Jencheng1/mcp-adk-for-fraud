#!/usr/bin/env python3
"""
GNN Model for Credit Card Fraud Detection

This script implements a Graph Neural Network (GNN) model for detecting fraud patterns
in credit card transactions using PyTorch Geometric.
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import GATConv, SAGEConv, GINConv, HeteroConv, global_mean_pool, global_max_pool
from torch_geometric.loader import DataLoader
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split

class GNNFraudDetection(nn.Module):
    """Graph Neural Network for fraud detection"""
    
    def __init__(self, hidden_channels=64, num_layers=3):
        super(GNNFraudDetection, self).__init__()
        
        # Node embedding layers
        self.user_embedding = nn.Linear(8, hidden_channels)
        self.card_embedding = nn.Linear(6, hidden_channels)
        self.transaction_embedding = nn.Linear(12, hidden_channels)
        self.merchant_embedding = nn.Linear(6, hidden_channels)
        self.location_embedding = nn.Linear(5, hidden_channels)
        self.device_embedding = nn.Linear(5, hidden_channels)
        
        # GNN layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('user', 'owns', 'card'): GATConv(hidden_channels, hidden_channels),
                ('card', 'made', 'transaction'): GATConv(hidden_channels, hidden_channels),
                ('transaction', 'at', 'merchant'): GATConv(hidden_channels, hidden_channels),
                ('transaction', 'in', 'location'): GATConv(hidden_channels, hidden_channels),
                ('transaction', 'using', 'device'): GATConv(hidden_channels, hidden_channels),
                ('transaction', 'followed_by', 'transaction'): GATConv(hidden_channels, hidden_channels),
                # Reverse edges
                ('card', 'owned_by', 'user'): GATConv(hidden_channels, hidden_channels),
                ('transaction', 'made_with', 'card'): GATConv(hidden_channels, hidden_channels),
                ('merchant', 'receives', 'transaction'): GATConv(hidden_channels, hidden_channels),
                ('location', 'has', 'transaction'): GATConv(hidden_channels, hidden_channels),
                ('device', 'used_in', 'transaction'): GATConv(hidden_channels, hidden_channels),
                ('transaction', 'preceded', 'transaction'): GATConv(hidden_channels, hidden_channels),
            })
            self.convs.append(conv)
        
        # Pooling layer
        self.pool = nn.Linear(hidden_channels * 2, hidden_channels)
        
        # Output layer for transaction nodes
        self.transaction_classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, data):
        # Get node features
        x_dict = {
            'user': self.user_embedding(data['user'].x),
            'card': self.card_embedding(data['card'].x),
            'transaction': self.transaction_embedding(data['transaction'].x),
            'merchant': self.merchant_embedding(data['merchant'].x),
            'location': self.location_embedding(data['location'].x),
            'device': self.device_embedding(data['device'].x)
        }
        
        # Message passing layers
        for conv in self.convs:
            x_dict = conv(x_dict, data.edge_index_dict)
            # Apply ReLU and dropout
            for node_type in x_dict.keys():
                x_dict[node_type] = F.relu(x_dict[node_type])
                x_dict[node_type] = F.dropout(x_dict[node_type], p=0.2, training=self.training)
        
        # Global pooling for graph-level features
        transaction_embeddings = x_dict['transaction']
        
        # Mean and max pooling
        mean_pool = global_mean_pool(transaction_embeddings, data['transaction'].batch)
        max_pool = global_max_pool(transaction_embeddings, data['transaction'].batch)
        
        # Combine pooled representations
        pooled = torch.cat([mean_pool, max_pool], dim=1)
        pooled = self.pool(pooled)
        
        # Predict fraud probability for transaction nodes
        fraud_scores = self.transaction_classifier(x_dict['transaction'])
        
        return fraud_scores, pooled

def prepare_node_features(data_dir):
    """Prepare node features from CSV files"""
    print("Preparing node features...")
    
    # Load data
    users_df = pd.read_csv(os.path.join(data_dir, 'users.csv'))
    cards_df = pd.read_csv(os.path.join(data_dir, 'cards.csv'))
    transactions_df = pd.read_csv(os.path.join(data_dir, 'all_transactions.csv'))
    merchants_df = pd.read_csv(os.path.join(data_dir, 'merchants.csv'))
    locations_df = pd.read_csv(os.path.join(data_dir, 'locations.csv'))
    devices_df = pd.read_csv(os.path.join(data_dir, 'devices.csv'))
    
    # Prepare user features
    user_features = users_df[['risk_score', 'fraud_history', 'kyc_verified']].copy()
    # One-hot encode segment
    segment_dummies = pd.get_dummies(users_df['segment'], prefix='segment')
    user_features = pd.concat([user_features, segment_dummies], axis=1)
    # Convert boolean columns to float
    user_features['fraud_history'] = user_features['fraud_history'].astype(float)
    user_features['kyc_verified'] = user_features['kyc_verified'].astype(float)
    
    # Prepare card features
    card_features = cards_df[['is_active', 'is_blocked', 'credit_limit', 'available_credit']].copy()
    # One-hot encode card type
    card_type_dummies = pd.get_dummies(cards_df['card_type'], prefix='card_type')
    card_features = pd.concat([card_features, card_type_dummies], axis=1)
    # Convert boolean columns to float
    card_features['is_active'] = card_features['is_active'].astype(float)
    card_features['is_blocked'] = card_features['is_blocked'].astype(float)
    # Normalize credit limit and available credit
    card_features['credit_limit'] = card_features['credit_limit'] / card_features['credit_limit'].max()
    card_features['available_credit'] = card_features['available_credit'] / card_features['credit_limit'].max()
    
    # Prepare transaction features
    transaction_features = transactions_df[['amount', 'is_online', 'is_fraudulent', 'fraud_score']].copy()
    # One-hot encode transaction type and payment method
    tx_type_dummies = pd.get_dummies(transactions_df['transaction_type'], prefix='tx_type')
    payment_dummies = pd.get_dummies(transactions_df['payment_method'], prefix='payment')
    transaction_features = pd.concat([transaction_features, tx_type_dummies, payment_dummies], axis=1)
    # Convert boolean columns to float
    transaction_features['is_online'] = transaction_features['is_online'].astype(float)
    transaction_features['is_fraudulent'] = transaction_features['is_fraudulent'].astype(float)
    # Normalize amount
    transaction_features['amount'] = transaction_features['amount'] / transaction_features['amount'].max()
    
    # Prepare merchant features
    merchant_features = merchants_df[['risk_score', 'fraud_rate', 'avg_transaction_amount', 'is_high_risk']].copy()
    # One-hot encode category
    category_dummies = pd.get_dummies(merchants_df['category'], prefix='category')
    merchant_features = pd.concat([merchant_features, category_dummies], axis=1)
    # Convert boolean columns to float
    merchant_features['is_high_risk'] = merchant_features['is_high_risk'].astype(float)
    # Normalize amount
    merchant_features['avg_transaction_amount'] = merchant_features['avg_transaction_amount'] / merchant_features['avg_transaction_amount'].max()
    
    # Prepare location features
    location_features = locations_df[['latitude', 'longitude', 'risk_score']].copy()
    # One-hot encode country
    country_dummies = pd.get_dummies(locations_df['country'], prefix='country')
    location_features = pd.concat([location_features, country_dummies], axis=1)
    # Normalize coordinates
    location_features['latitude'] = (location_features['latitude'] - location_features['latitude'].min()) / (location_features['latitude'].max() - location_features['latitude'].min())
    location_features['longitude'] = (location_features['longitude'] - location_features['longitude'].min()) / (location_features['longitude'].max() - location_features['longitude'].min())
    
    # Prepare device features
    device_features = devices_df[['is_mobile', 'is_known_device', 'risk_score']].copy()
    # One-hot encode device type
    device_type_dummies = pd.get_dummies(devices_df['device_type'], prefix='device_type')
    device_features = pd.concat([device_features, device_type_dummies], axis=1)
    # Convert boolean columns to float
    device_features['is_mobile'] = device_features['is_mobile'].astype(float)
    device_features['is_known_device'] = device_features['is_known_device'].astype(float)
    
    # Create node mappings (original ID to index)
    user_mapping = {user_id: idx for idx, user_id in enumerate(users_df['user_id'])}
    card_mapping = {card_id: idx for idx, card_id in enumerate(cards_df['card_id'])}
    transaction_mapping = {tx_id: idx for idx, tx_id in enumerate(transactions_df['transaction_id'])}
    merchant_mapping = {merch_id: idx for idx, merch_id in enumerate(merchants_df['merchant_id'])}
    location_mapping = {loc_id: idx for idx, loc_id in enumerate(locations_df['location_id'])}
    device_mapping = {dev_id: idx for idx, dev_id in enumerate(devices_df['device_id'])}
    
    # Convert features to PyTorch tensors
    user_features_tensor = torch.tensor(user_features.values, dtype=torch.float)
    card_features_tensor = torch.tensor(card_features.values, dtype=torch.float)
    transaction_features_tensor = torch.tensor(transaction_features.values, dtype=torch.float)
    merchant_features_tensor = torch.tensor(merchant_features.values, dtype=torch.float)
    location_features_tensor = torch.tensor(location_features.values, dtype=torch.float)
    device_features_tensor = torch.tensor(device_features.values, dtype=torch.float)
    
    # Extract labels
    labels = transaction_features['is_fraudulent'].values
    
    return {
        'user_features': user_features_tensor,
        'card_features': card_features_tensor,
        'transaction_features': transaction_features_tensor,
        'merchant_features': merchant_features_tensor,
        'location_features': location_features_tensor,
        'device_features': device_features_tensor,
        'user_mapping': user_mapping,
        'card_mapping': card_mapping,
        'transaction_mapping': transaction_mapping,
        'merchant_mapping': merchant_mapping,
        'location_mapping': location_mapping,
        'device_mapping': device_mapping,
        'labels': labels,
        'users_df': users_df,
        'cards_df': cards_df,
        'transactions_df': transactions_df,
        'merchants_df': merchants_df,
        'locations_df': locations_df,
        'devices_df': devices_df
    }

def create_heterogeneous_graph(data_dict):
    """Create a heterogeneous graph from the prepared data"""
    print("Creating heterogeneous graph...")
    
    # Extract data
    users_df = data_dict['users_df']
    cards_df = data_dict['cards_df']
    transactions_df = data_dict['transactions_df']
    merchants_df = data_dict['merchants_df']
    locations_df = data_dict['locations_df']
    devices_df = data_dict['devices_df']
    
    user_mapping = data_dict['user_mapping']
    card_mapping = data_dict['card_mapping']
    transaction_mapping = data_dict['transaction_mapping']
    merchant_mapping = data_dict['merchant_mapping']
    location_mapping = data_dict['location_mapping']
    device_mapping = data_dict['device_mapping']
    
    # Create heterogeneous graph
    data = HeteroData()
    
    # Add node features
    data['user'].x = data_dict['user_features']
    data['card'].x = data_dict['card_features']
    data['transaction'].x = data_dict['transaction_features']
    data['merchant'].x = data_dict['merchant_features']
    data['location'].x = data_dict['location_features']
    data['device'].x = data_dict['device_features']
    
    # Add node labels (for transactions)
    data['transaction'].y = torch.tensor(data_dict['labels'], dtype=torch.float)
    
    # Create edges
    
    # User-Card edges (OWNS)
    user_card_src = [user_mapping[user_id] for user_id in cards_df['user_id']]
    user_card_dst = [card_mapping[card_id] for card_id in cards_df['card_id']]
    data['user', 'owns', 'card'].edge_index = torch.tensor([user_card_src, user_card_dst], dtype=torch.long)
    
    # Card-Transaction edges (MADE)
    card_tx_src = []
    card_tx_dst = []
    for _, tx in transactions_df.iterrows():
        if tx['card_id'] in card_mapping:
            card_tx_src.append(card_mapping[tx['card_id']])
            card_tx_dst.append(transaction_mapping[tx['transaction_id']])
    data['card', 'made', 'transaction'].edge_index = torch.tensor([card_tx_src, card_tx_dst], dtype=torch.long)
    
    # Transaction-Merchant edges (AT)
    tx_merchant_src = []
    tx_merchant_dst = []
    for _, tx in transactions_df.iterrows():
        if tx['merchant_id'] in merchant_mapping:
            tx_merchant_src.append(transaction_mapping[tx['transaction_id']])
            tx_merchant_dst.append(merchant_mapping[tx['merchant_id']])
    data['transaction', 'at', 'merchant'].edge_index = torch.tensor([tx_merchant_src, tx_merchant_dst], dtype=torch.long)
    
    # Transaction-Location edges (IN)
    tx_location_src = []
    tx_location_dst = []
    for _, tx in transactions_df.iterrows():
        if tx['location_id'] in location_mapping:
            tx_location_src.append(transaction_mapping[tx['transaction_id']])
            tx_location_dst.append(location_mapping[tx['location_id']])
    data['transaction', 'in', 'location'].edge_index = torch.tensor([tx_location_src, tx_location_dst], dtype=torch.long)
    
    # Transaction-Device edges (USING)
    tx_device_src = []
    tx_device_dst = []
    for _, tx in transactions_df.iterrows():
        if pd.notna(tx['device_id']) and tx['device_id'] in device_mapping:
            tx_device_src.append(transaction_mapping[tx['transaction_id']])
            tx_device_dst.append(device_mapping[tx['device_id']])
    data['transaction', 'using', 'device'].edge_index = torch.tensor([tx_device_src, tx_device_dst], dtype=torch.long)
    
    # Transaction-Transaction edges (FOLLOWED_BY)
    # Sort transactions by timestamp for each card
    tx_followed_by_src = []
    tx_followed_by_dst = []
    
    for card_id in cards_df['card_id']:
        card_txs = transactions_df[transactions_df['card_id'] == card_id].sort_values('timestamp')
        if len(card_txs) > 1:
            for i in range(len(card_txs) - 1):
                tx_followed_by_src.append(transaction_mapping[card_txs.iloc[i]['transaction_id']])
                tx_followed_by_dst.append(transaction_mapping[card_txs.iloc[i+1]['transaction_id']])
    
    if tx_followed_by_src:  # Only add if there are edges
        data['transaction', 'followed_by', 'transaction'].edge_index = torch.tensor([tx_followed_by_src, tx_followed_by_dst], dtype=torch.long)
    
    # Add reverse edges
    data = add_reverse_edges(data)
    
    return data

def add_reverse_edges(data):
    """Add reverse edges to the heterogeneous graph"""
    # User-Card reverse
    data['card', 'owned_by', 'user'].edge_index = data['user', 'owns', 'card'].edge_index.flip([0])
    
    # Card-Transaction reverse
    data['transaction', 'made_with', 'card'].edge_index = data['card', 'made', 'transaction'].edge_index.flip([0])
    
    # Transaction-Merchant reverse
    data['merchant', 'receives', 'transaction'].edge_index = data['transaction', 'at', 'merchant'].edge_index.flip([0])
    
    # Transaction-Location reverse
    data['location', 'has', 'transaction'].edge_index = data['transaction', 'in', 'location'].edge_index.flip([0])
    
    # Transaction-Device reverse
    if 'transaction__using__device' in data.edge_types:
        data['device', 'used_in', 'transaction'].edge_index = data['transaction', 'using', 'device'].edge_index.flip([0])
    
    # Transaction-Transaction reverse
    if 'transaction__followed_by__transaction' in data.edge_types:
        data['transaction', 'preceded', 'transaction'].edge_index = data['transaction', 'followed_by', 'transaction'].edge_index.flip([0])
    
    return data

def train_gnn_model(data, epochs=50, lr=0.001, batch_size=32):
    """Train the GNN model"""
    print("Training GNN model...")
    
    # Split data into train/val/test
    num_transactions = data['transaction'].x.size(0)
    indices = list(range(num_transactions))
    
    # Use stratified sampling to handle class imbalance
    labels = data['transaction'].y.numpy()
    train_idx, test_idx = train_test_split(indices, test_size=0.2, stratify=labels, random_state=42)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.25, stratify=labels[train_idx], random_state=42)
    
    # Create masks
    train_mask = torch.zeros(num_transactions, dtype=torch.bool)
    val_mask = torch.zeros(num_transactions, dtype=torch.bool)
    test_mask = torch.zeros(num_transactions, dtype=torch.bool)
    
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    
    data['transaction'].train_mask = train_mask
    data['transaction'].val_mask = val_mask
    data['transaction'].test_mask = test_mask
    
    # Initialize model
    model = GNNFraudDetection()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    # Training loop
    best_val_auc = 0
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        fraud_scores, _ = model(data)
        
        # Calculate loss on training set
        train_loss = criterion(fraud_scores[train_mask].squeeze(), data['transaction'].y[train_mask])
        
        # Backward pass
        train_loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            fraud_scores, _ = model(data)
            val_loss = criterion(fraud_scores[val_mask].squeeze(), data['transaction'].y[val_mask])
            
            # Calculate metrics
            val_preds = (fraud_scores[val_mask].squeeze() > 0.5).float().numpy()
            val_labels = data['transaction'].y[val_mask].numpy()
            val_scores = fraud_scores[val_mask].squeeze().numpy()
            
            val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(
                val_labels, val_preds, average='binary'
            )
            val_auc = roc_auc_score(val_labels, val_scores)
        
        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss.item():.4f}, "
                  f"Val Loss: {val_loss.item():.4f}, Val AUC: {val_auc:.4f}, "
                  f"Val F1: {val_f1:.4f}, Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}")
        
        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict()
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        fraud_scores, _ = model(data)
        test_loss = criterion(fraud_scores[test_mask].squeeze(), data['transaction'].y[test_mask])
        
        # Calculate metrics
        test_preds = (fraud_scores[test_mask].squeeze() > 0.5).float().numpy()
        test_labels = data['transaction'].y[test_mask].numpy()
        test_scores = fraud_scores[test_mask].squeeze().numpy()
        
        test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
            test_labels, test_preds, average='binary'
        )
        test_auc = roc_auc_score(test_labels, test_scores)
    
    print("\nTest Results:")
    print(f"Test Loss: {test_loss.item():.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Test F1: {test_f1:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    
    return model, {
        'test_auc': test_auc,
        'test_f1': test_f1,
        'test_precision': test_precision,
        'test_recall': test_recall
    }

def save_model(model, output_dir):
    """Save the trained model"""
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, 'gnn_fraud_model.pt')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

def main():
    parser = argparse.ArgumentParser(description='GNN for Credit Card Fraud Detection')
    parser.add_argument('--data-dir', type=str, default='/home/ubuntu/credit_card_fraud_detection/data',
                        help='Directory containing data CSV files')
    parser.add_argument('--output-dir', type=str, default='/home/ubuntu/credit_card_fraud_detection/src/gnn/models',
                        help='Directory to save the trained model')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    
    args = parser.parse_args()
    
    # Prepare data
    data_dict = prepare_node_features(args.data_dir)
    
    # Create heterogeneous graph
    graph_data = create_heterogeneous_graph(data_dict)
    
    # Train model
    model, metrics = train_gnn_model(
        graph_data,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size
    )
    
    # Save model
    save_model(model, args.output_dir)
    
    # Save metrics
    metrics_path = os.path.join(args.output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f)
    
    print(f"Metrics saved to {metrics_path}")

if __name__ == "__main__":
    main()

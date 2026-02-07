import pytest
import torch
import sys
import os

# Add parent dir to path to allow imports if running from ai_engine/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_engine.model import LightGCN
from ai_engine.loss import BPRLoss
from ai_engine.train import train_epoch

def test_model_learning_capability():
    """
    TDD Test: Verify that the model actually learns using BPR Loss.
    """
    # 1. Setup minimal data
    num_users = 3
    num_items = 2
    embedding_dim = 4
    
    # User 0->Item 0
    # User 1->Item 1
    # User 2->Item 0
    # Items are indexed 0,1. In LightGCN graph, they are offset by num_users.
    # item_0_idx = 3, item_1_idx = 4
    
    train_edge_index = torch.tensor([
        [0, 1, 2],    
        [3, 4, 3]     
    ])
    
    # Make undirected for message passing
    edge_index = torch.cat([
        train_edge_index, 
        torch.stack([train_edge_index[1], train_edge_index[0]], dim=0)
    ], dim=1)
    
    model = LightGCN(num_users, num_items, embedding_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    bpr_loss = BPRLoss()
    
    # Snapshot initial weights
    initial_user_weights = model.user_emb.weight.clone()
    
    # 2. Run Training Step
    loss = train_epoch(
        model, optimizer, bpr_loss,
        edge_index, train_edge_index,
        num_users, num_items,
        batch_size=2
    )
    
    # 3. Assertions
    print(f"Training Loss: {loss}")
    
    # Loss should be a valid number
    assert loss > 0
    
    # Weights MUST change
    new_user_weights = model.user_emb.weight
    assert not torch.allclose(initial_user_weights, new_user_weights), "Weights did not update! Optimization failed."
    
    # Check shape of embeddings
    assert new_user_weights.shape == (num_users, embedding_dim)

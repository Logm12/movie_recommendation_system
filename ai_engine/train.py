import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from model import LightGCN
from loss import BPRLoss
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance
import os
import random

# Config
DATA_PATH = "../data/ml-latest-small/ratings.csv"
EMBEDDING_DIM = 64
EPOCHS = 10
LR = 0.001
BATCH_SIZE = 1024

def load_data(path=DATA_PATH):
    print(f"Loading data from {path}...")
    df = pd.read_csv(path)
    
    # ID Remapping
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    
    df['user_idx'] = user_encoder.fit_transform(df['userId'])
    df['item_idx'] = item_encoder.fit_transform(df['movieId'])
    
    num_users = df['user_idx'].nunique()
    num_items = df['item_idx'].nunique()
    
    print(f"Num Users: {num_users}, Num Items: {num_items}")
    
    # Build Edge Index (User->Item)
    # PyG expects [2, num_edges]
    # We offset item indices by num_users to create a single graph
    src = torch.tensor(df['user_idx'].values)
    dst = torch.tensor(df['item_idx'].values) + num_users
    
    # Keep only User->Item for sampling positive pairs
    # (src contains user indices 0..num_users-1)
    train_edge_index = torch.stack([src, dst], dim=0)
    
    # For Message Passing, we need undirected (bi-directional)
    edge_index = torch.cat([train_edge_index, torch.stack([dst, src], dim=0)], dim=1)
    
    return df, edge_index, train_edge_index, num_users, num_items, user_encoder, item_encoder

def sample_batch(train_edge_index, num_users, num_items, batch_size=BATCH_SIZE):
    """
    Sample a batch of (user, pos_item, neg_item) triplets.
    """
    total_edges = train_edge_index.shape[1]
    
    # Randomly select indices
    indices = torch.randint(0, total_edges, (batch_size,))
    batch_edges = train_edge_index[:, indices]
    
    users = batch_edges[0]
    pos_items = batch_edges[1]
    
    # Negative Sampling
    # Heuristic: Randomly select an item. Collision probability is low for sparse data.
    # In production, check if (u, j) exists.
    neg_items = torch.randint(0, num_items, (batch_size,)) + num_users
    
    return users, pos_items, neg_items

def train_epoch(model, optimizer, bpr_loss, edge_index, train_edge_index, num_users, num_items, batch_size=BATCH_SIZE):
    """
    Train a single epoch using BPR Loss.
    """
    model.train()
    total_loss = 0
    num_batches = (train_edge_index.shape[1] + batch_size - 1) // batch_size
    
    # Shuffle edges for training
    # For simplicity in this implementation, we just random sample 'num_batches' times
    # A strict epoch iterates all edges.
    
    for _ in range(num_batches):
        optimizer.zero_grad()
        
        # 1. Sample Batch
        users, pos_items, neg_items = sample_batch(train_edge_index, num_users, num_items, batch_size)
        
        # 2. Forward Pass (Get all embeddings)
        # We need generic embeddings from the graph propagation
        all_users_emb, all_items_emb = model(edge_index)
        
        # 3. Concatenate for easy indexing (0..num_users+num_items)
        all_embs = torch.cat([all_users_emb, all_items_emb], dim=0)
        
        # 4. Lookup specific embeddings for the batch
        # Note: pos_items/neg_items are already offset by num_users
        batch_users_emb = all_embs[users]
        batch_pos_emb = all_embs[pos_items]
        batch_neg_emb = all_embs[neg_items]
        
        # 5. Look up Initial Embeddings (Layer 0) for Regularization
        # LightGCN regularizes E^0 
        user_emb0 = model.user_emb(users)
        pos_emb0 = model.item_emb(pos_items - num_users) # Remove offset for Embedding layer
        neg_emb0 = model.item_emb(neg_items - num_users)
        
        # 6. Compute Loss
        loss = bpr_loss(
            batch_users_emb, batch_pos_emb, batch_neg_emb,
            user_emb0, pos_emb0, neg_emb0
        )
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / num_batches

def train():
    df, edge_index, train_edge_index, num_users, num_items, user_encoder, item_encoder = load_data()
    
    device = torch.device('cpu') 
    model = LightGCN(num_users, num_items, EMBEDDING_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    bpr_loss = BPRLoss()
    
    print(f"Starting training for {EPOCHS} epochs...")
    
    for epoch in range(EPOCHS):
        loss = train_epoch(
            model, optimizer, bpr_loss, 
            edge_index, train_edge_index, 
            num_users, num_items
        )
        if epoch % 1 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss:.4f}")
            
    print("Training finished.")
    
    # Extract Embeddings
    model.eval()
    with torch.no_grad():
        final_users, final_items = model(edge_index)
    
    # --- Indexing to Qdrant ---
    print("Indexing to Qdrant...")
    try:
        client = QdrantClient("localhost", port=6333)
        collection_name = "movies"
        
        # Recreate collection
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        
        # Batch Upload
        points = []
        for idx, original_id in enumerate(item_encoder.classes_):
            vector = final_items[idx].tolist()
            points.append(PointStruct(
                id=int(original_id), 
                vector=vector,
                payload={"movie_id": int(original_id)}
            ))
        
        # Batch upsert (chunking 100 points)
        batch_size = 100
        for i in range(0, len(points), batch_size):
            client.upsert(
                collection_name=collection_name,
                points=points[i:i+batch_size]
            )
        print(f"Indexed {len(points)} movies.")
    except Exception as e:
        print(f"Warning: Qdrant indexing failed (is it running?): {e}")
        print("Skipping indexing, but saving local embeddings.")
    
    # Save User Embeddings for Backend
    torch.save({
        'user_embeddings': final_users,
        'user_encoder': user_encoder
    }, "user_embeddings.pt")
    print("User embeddings saved.")

if __name__ == "__main__":
    train()

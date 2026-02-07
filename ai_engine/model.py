import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree

class LightGCN(MessagePassing):
    def __init__(self, num_users, num_items, embedding_dim=64, num_layers=3):
        super().__init__(aggr='add')
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        
        self.user_emb = torch.nn.Embedding(num_users, embedding_dim)
        self.item_emb = torch.nn.Embedding(num_items, embedding_dim)
        
        # Initialize weights
        torch.nn.init.normal_(self.user_emb.weight, std=0.1)
        torch.nn.init.normal_(self.item_emb.weight, std=0.1)
        
    def forward(self, edge_index):
        edge_index, norm = self.compute_normalization(edge_index)
        
        # Step 1: Initial Embeddings (Layer 0)
        x = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        embs = [x]
        
        # Step 2: Message Passing (Layers 1..N)
        for _ in range(self.num_layers):
            x = self.propagate(edge_index, x=x, norm=norm)
            embs.append(x)
            
        # Step 3: Final Embedding (Mean/Sum)
        final_emb = torch.stack(embs, dim=1).mean(dim=1)
        
        # Split back into users and items
        users, items = torch.split(final_emb, [self.num_users, self.num_items])
        return users, items
    
    def compute_normalization(self, edge_index):
        # LightGCN uses symmetric normalization: D^(-0.5) * A * D^(-0.5)
        # Here we just implement the simple version supported by pyG or custom
        # For simplicity in this demo, we trust standard propagation.
        # But efficiently: row-normalize or symmetric normalize
        
        from torch_geometric.utils import add_self_loops, degree
        row, col = edge_index
        deg = degree(col, self.num_users + self.num_items)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        return edge_index, norm

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

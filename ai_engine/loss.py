import torch
import torch.nn as nn
import torch.nn.functional as F

class BPRLoss(nn.Module):
    """
    Bayesian Personalized Ranking Loss for recommender systems.
    L = -sum(log(sigmoid(pos_score - neg_score))) + lambda * L2_norm
    """
    def __init__(self, lambda_reg=1e-4):
        super(BPRLoss, self).__init__()
        self.lambda_reg = lambda_reg

    def forward(self, users_emb, pos_items_emb, neg_items_emb, 
                user_emb0, pos_emb0, neg_emb0):
        """
        Args:
            users_emb: Deep embeddings of users (after LightGCN)
            pos_items_emb: Deep embeddings of positive items
            neg_items_emb: Deep embeddings of negative items
            user_emb0: Initial embeddings of users (Layer 0) for regularization
            pos_emb0: Initial embeddings of positive items (Layer 0)
            neg_emb0: Initial embeddings of negative items (Layer 0)
        """
        # BPR Loss
        pos_scores = torch.mul(users_emb, pos_items_emb).sum(dim=1)
        neg_scores = torch.mul(users_emb, neg_items_emb).sum(dim=1)
        
        loss = -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()
        
        # Regularization (L2 Norm on initial embeddings)
        reg_loss = (1/2) * (user_emb0.norm(2).pow(2) + 
                            pos_emb0.norm(2).pow(2) +
                            neg_emb0.norm(2).pow(2)) / float(users_emb.shape[0])
                            
        return loss + self.lambda_reg * reg_loss

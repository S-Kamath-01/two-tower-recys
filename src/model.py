"""
model.py

Two-Tower Neural Recommender — model definition.

Architecture:
    - Two independent towers (user, item), each: Embedding -> Linear -> ReLU -> Linear
    - Shared output dimension `embed_dim` so the dot product between the two
      towers' outputs is well-defined (same coordinate space, same units)
    - Scoring is dot-product only — no concatenation, no MLP combining the
      two towers. This is the whole point of "Two-Tower": item embeddings
      can be precomputed once, independent of any user, and indexed with
      something like FAISS for fast approximate nearest-neighbor retrieval
      at serving time. An MLP ranking head would break that property, since
      it would require a forward pass for every (user, item) pair.
"""

import torch
import torch.nn as nn


class Tower(nn.Module):
    """
    A single tower: maps an integer ID -> dense vector in the shared
    embedding space.

    nn.Embedding(vocab_size, embed_dim) is a raw lookup table. Used alone,
    this tower would be mathematically identical to matrix factorization.
    The Linear -> ReLU -> Linear projection on top is what gives the tower
    nonlinear capacity to reshape that raw embedding before scoring — it's
    what makes this a *neural* two-tower model rather than MF with extra
    bookkeeping.
    """

    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int = None,
                 dropout: float = 0.0):
        super().__init__()
        hidden_dim = hidden_dim or embed_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # Default nn.Embedding init is N(0, 1), which makes initial dot
        # products unnecessarily large and the early loss landscape noisy.
        # A smaller std gives more stable early training.
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.1)

        self.projection = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """
        ids: (B,) integer tensor of user_id or movie_id
        returns: (B, embed_dim)
        """
        raw = self.embedding(ids)      # (B, embed_dim) — table lookup
        out = self.projection(raw)     # (B, embed_dim) — nonlinear reshape
        return out


class TwoTowerModel(nn.Module):
    """
    Two-Tower retrieval model.

    forward() is for training: takes (user, pos_item, neg_item) triplets
    and returns the two dot-product scores the BPR loss needs.

    get_user_embeddings() / get_item_embeddings() are for inference: each
    tower can be called independently, so item embeddings can be
    precomputed once and indexed for retrieval without ever touching the
    user tower.
    """

    def __init__(self, num_users: int, num_movies: int, embed_dim: int = 64,
                 hidden_dim: int = None, dropout: float = 0.0):
        super().__init__()
        self.user_tower = Tower(num_users, embed_dim, hidden_dim, dropout)
        self.item_tower = Tower(num_movies, embed_dim, hidden_dim, dropout)

    def forward(self, user_ids: torch.Tensor, pos_item_ids: torch.Tensor,
                neg_item_ids: torch.Tensor):
        """
        user_ids, pos_item_ids, neg_item_ids: each (B,)

        returns:
            pos_scores: (B,) — dot(user_emb, pos_item_emb) per row
            neg_scores: (B,) — dot(user_emb, neg_item_emb) per row
        """
        user_emb = self.user_tower(user_ids)       # (B, d)
        pos_emb = self.item_tower(pos_item_ids)    # (B, d)
        neg_emb = self.item_tower(neg_item_ids)    # (B, d)

        pos_scores = (user_emb * pos_emb).sum(dim=1)   # (B,)
        neg_scores = (user_emb * neg_emb).sum(dim=1)   # (B,)

        return pos_scores, neg_scores

    def get_user_embeddings(self, user_ids: torch.Tensor) -> torch.Tensor:
        """(B,) -> (B, embed_dim). Used at inference / index-build time."""
        return self.user_tower(user_ids)

    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        """(B,) -> (B, embed_dim). Used at inference / index-build time."""
        return self.item_tower(item_ids)


if __name__ == "__main__":
    # Quick shape sanity check against your actual dataset stats.
    NUM_USERS = 6037
    NUM_MOVIES = 3533
    EMBED_DIM = 64
    BATCH_SIZE = 32

    model = TwoTowerModel(num_users=NUM_USERS, num_movies=NUM_MOVIES, embed_dim=EMBED_DIM)

    user_ids = torch.randint(0, NUM_USERS, (BATCH_SIZE,))
    pos_ids = torch.randint(0, NUM_MOVIES, (BATCH_SIZE,))
    neg_ids = torch.randint(0, NUM_MOVIES, (BATCH_SIZE,))

    pos_scores, neg_scores = model(user_ids, pos_ids, neg_ids)
    print(f"pos_scores shape: {pos_scores.shape}")    # expect (32,)
    print(f"neg_scores shape: {neg_scores.shape}")    # expect (32,)

    user_emb = model.get_user_embeddings(user_ids)
    item_emb = model.get_item_embeddings(pos_ids)
    print(f"user_emb shape: {user_emb.shape}")         # expect (32, 64)
    print(f"item_emb shape: {item_emb.shape}")         # expect (32, 64)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total trainable parameters: {num_params:,}")
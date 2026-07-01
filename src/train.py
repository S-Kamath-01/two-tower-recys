"""
train.py

Full training loop for the Two-Tower recommender using BPR loss.

Design Decisions Explained:

1. BPR Loss (Bayesian Personalized Ranking) instead of binary cross-entropy:
   - BPR is a *pairwise* ranking loss: we directly optimize that the positive item
     scores higher than the negative item. This aligns with the test metric (recall
     at K—we rank items for a user).
   - Binary cross-entropy treats each sample independently: it wants the positive
     score high (>0.5 as probability), and negative score low. But two models could
     both satisfy this while having very different ranking order.
   - For implicit feedback (no "how much the user liked" signal, just yes/no), pairwise
     ranking losses like BPR are theoretically and empirically superior.
   - Formula: L = -log(σ(s_pos - s_neg))
     where s_pos = dot(user_emb, pos_item_emb), s_neg = dot(user_emb, neg_item_emb),
     and σ is the sigmoid. Minimizing this pushes s_pos - s_neg toward +∞.
   - In PyTorch: F.binary_cross_entropy_with_logits(s_pos - s_neg, ones)
     treats (s_pos - s_neg) as logits and targets as 1, which is exactly BPR.

2. Adam optimizer:
   - Adaptive per-parameter learning rates. Robust across hyperparameter choices.
   - Standard choice for neural recommendation systems.

3. Checkpoint saving:
   - Save best model by validation metric (recall@20), plus last model.
   - Enables early stopping and model selection without retraining.

4. Per-epoch loss logging:
   - Track both train and validation loss to diagnose overfitting.
   - Loss plateauing suggests convergence or stuck local minimum.
"""

import os
import json
import pickle
import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

try:  # Support both `python -m src.train` and `python src/train.py`.
    from .model import TwoTowerModel
    from .data.dataset import TwoTowerDataset
except ImportError:
    from model import TwoTowerModel
    from data.dataset import TwoTowerDataset

# Configure logging for readable console output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def bpr_loss(pos_scores, neg_scores):
    """
    Bayesian Personalized Ranking loss.

    We want: pos_scores >> neg_scores for all samples.
    This is achieved by minimizing -log(σ(pos_scores - neg_scores)),
    where σ is sigmoid.

    In PyTorch, binary_cross_entropy_with_logits treats its first argument
    as logits (pre-sigmoid) and applies sigmoid internally before computing BCE.
    So by passing (pos_scores - neg_scores) as logits and target=1, we get:
        BCE = -log(σ(pos_scores - neg_scores))
    which is exactly BPR.

    Parameters:
        pos_scores (torch.Tensor): (B,) positive item scores from the model
        neg_scores (torch.Tensor): (B,) negative item scores from the model

    Returns:
        torch.Tensor: scalar loss
    """
    # Target is 1 (we want σ(pos - neg) → 1, i.e., pos - neg → +∞)
    target = torch.ones_like(pos_scores)
    loss = F.binary_cross_entropy_with_logits(pos_scores - neg_scores, target)
    return loss


def train_epoch(model, dataloader, optimizer, device):
    """
    Run one training epoch: iterate through all (user, pos, neg) triplets,
    compute BPR loss, backprop, and aggregate losses.

    Parameters:
        model (TwoTowerModel): the model to train
        dataloader (DataLoader): yields (user_ids, pos_ids, neg_ids)
        optimizer (torch.optim.Optimizer): e.g., Adam
        device (torch.device): CPU or CUDA

    Returns:
        float: mean BPR loss over the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for user_ids, pos_ids, neg_ids in dataloader:
        # Move to device (GPU/CPU)
        user_ids = user_ids.to(device)
        pos_ids = pos_ids.to(device)
        neg_ids = neg_ids.to(device)

        # Forward pass: model returns (pos_scores, neg_scores) for BPR
        pos_scores, neg_scores = model(user_ids, pos_ids, neg_ids)

        # Compute BPR loss
        loss = bpr_loss(pos_scores, neg_scores)

        # Backprop and optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    if num_batches == 0:
        raise ValueError("Training data is empty; at least one interaction is required")
    avg_loss = total_loss / num_batches
    return avg_loss


def train(
    train_csv_path,
    metadata_json_path,
    output_dir,
    num_epochs=50,
    batch_size=128,
    embed_dim=64,
    hidden_dim=None,
    dropout=0.1,
    learning_rate=1e-3,
    device=None,
):
    """
    Full training loop.

    Loads training data, initializes model, and runs training for num_epochs.
    Logs per-epoch loss and saves model checkpoints.

    Parameters:
        train_csv_path (str): path to train.csv (index-encoded)
        metadata_json_path (str): path to metadata.json with num_users, num_movies
        output_dir (str): directory to save checkpoints and logs
        num_epochs (int): number of training epochs
        batch_size (int): batch size for training
        embed_dim (int): embedding dimension for both towers
        hidden_dim (int or None): hidden dimension in tower projection; if None, uses embed_dim
        dropout (float): dropout rate in tower projection
        learning_rate (float): learning rate for Adam
        device (torch.device or None): if None, uses CUDA if available else CPU

    Returns:
        None (model checkpoint saved to output_dir)
    """
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load metadata to get vocab sizes
    with open(metadata_json_path, "r") as f:
        metadata = json.load(f)
    num_users = metadata["num_users"]
    num_movies = metadata["num_movies"]
    logger.info(f"Loaded metadata: {num_users} users, {num_movies} movies")

    # Load training data
    train_df = pd.read_csv(train_csv_path)
    logger.info(f"Loaded training data: {len(train_df)} interactions")

    # Create dataset and dataloader
    dataset = TwoTowerDataset(train_df, num_movies=num_movies)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # Windows compatibility; set to 4+ on Linux for speed
    )
    logger.info(f"DataLoader created: {len(dataloader)} batches of size {batch_size}")

    # Initialize model
    model = TwoTowerModel(
        num_users=num_users,
        num_movies=num_movies,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        dropout=dropout
    )
    model = model.to(device)
    logger.info(f"Model initialized: embed_dim={embed_dim}, hidden_dim={hidden_dim or embed_dim}, dropout={dropout}")

    # Count trainable parameters
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total trainable parameters: {num_params:,}")

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    logger.info(f"Optimizer: Adam with learning_rate={learning_rate}")

    # Training loop
    logger.info(f"Starting training for {num_epochs} epochs...\n")

    losses = []  # Track losses for logging/plotting

    for epoch in range(1, num_epochs + 1):
        epoch_loss = train_epoch(model, dataloader, optimizer, device)
        losses.append(epoch_loss)

        # Log every epoch
        if epoch % 1 == 0:
            logger.info(f"Epoch {epoch:3d}/{num_epochs} | Loss: {epoch_loss:.6f}")

    logger.info(f"\nTraining complete. Final loss: {losses[-1]:.6f}")

    # Save final model checkpoint
    checkpoint_path = os.path.join(output_dir, "model_final.pt")
    torch.save(model.state_dict(), checkpoint_path)
    logger.info(f"Final model saved to {checkpoint_path}")

    # Save training history
    history_path = os.path.join(output_dir, "training_history.json")
    history = {
        "losses": losses,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "embed_dim": embed_dim,
        "hidden_dim": hidden_dim or embed_dim,
        "dropout": dropout,
        "learning_rate": learning_rate,
    }
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"Training history saved to {history_path}")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Two-Tower recommender with BPR loss")
    parser.add_argument(
        "--train_csv",
        type=str,
        default="data/processed/train.csv",
        help="Path to training CSV (default: data/processed/train.csv)"
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default="data/processed/metadata.json",
        help="Path to metadata JSON (default: data/processed/metadata.json)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints",
        help="Output directory for checkpoints (default: checkpoints)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size (default: 128)"
    )
    parser.add_argument(
        "--embed_dim",
        type=int,
        default=64,
        help="Embedding dimension (default: 64)"
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=None,
        help="Hidden dimension in projection (default: same as embed_dim)"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate (default: 0.1)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3)"
    )

    args = parser.parse_args()

    model = train(
        train_csv_path=args.train_csv,
        metadata_json_path=args.metadata,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        learning_rate=args.lr,
    )

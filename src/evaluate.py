"""
evaluate.py

Evaluation metrics for the Two-Tower recommender on the temporal test set.

Design Decisions Explained:

1. Recall@K, Hit Rate@K, MRR@K why these metrics?
   - Recall@K: fraction of relevant items (user's test positive) that appear in top-K predictions.
     Directly measures coverage of user preferences in the recommendation list.
   - Hit Rate@K: binary indicator whether the user's test positive is in top-K.
     Simpler than Recall (always 0 or 1 per user), but aggregates well across users.
   - MRR@K (Mean Reciprocal Rank): average of (1 / rank) for the test positive.
     Penalizes incorrect ranking—a relevant item at rank 2 scores 0.5, at rank 10 scores 0.1.
   - These are standard for implicit feedback recommendation tasks (see RecSys literature).

2. Temporal test set evaluation:
   - Test set has exactly 1 positive per user (most recent interaction).
   - We rank all movies for that user and measure if the test positive appears in top-K.
   - This mimics real-world deployment: given a user, return top-K items to recommend.

3. Per-user ranking via dot-product:
   - For each test user, compute dot(user_emb, all_movie_embs).
   - This is efficient (one forward pass per user + matrix multiply).
   - Alternative: FAISS ANN would be faster at scale, but exact ranking is fine for evaluation.

4. No training set contamination:
   - user_positives (from train.csv) excludes test interactions.
   - So we never recommend items the user already interacted with in training.
   - This ensures clean evaluation (not rewarding the model for replicating the past).
"""

import os
import json
import pickle
import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import pandas as pd
import numpy as np

try:  # Support package and direct-script execution.
    from .model import TwoTowerModel
except ImportError:
    from model import TwoTowerModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def recall_at_k(predictions, positives, k):
    """
    Recall@K: fraction of relevant items that appear in top-K predictions.

    For each user:
        recall_k = |{predicted_k} ∩ {positives}| / |{positives}|

    Since test set has exactly 1 positive per user:
        recall_k = 1 if positive in top-K, else 0

    Aggregates across all users via mean.

    Parameters:
        predictions (dict): {user_id: [top_k_movie_ids]}
        positives (dict): {user_id: set of relevant movie_ids}
        k (int): cutoff rank

    Returns:
        float: mean recall@k across all users
    """
    recalls = []
    for user_id, pred_movies in predictions.items():
        if user_id not in positives:
            continue
        relevant = positives[user_id]
        # Top-k predictions
        top_k = set(pred_movies[:k])
        # Recall = intersection / |relevant|
        recall = len(top_k & relevant) / len(relevant)
        recalls.append(recall)

    return np.mean(recalls) if recalls else 0.0


def hit_rate_at_k(predictions, positives, k):
    """
    Hit Rate@K: fraction of users for whom the top-K list contains a relevant item.

    For each user:
        hit_k = 1 if any positive in top-K, else 0

    Aggregates via mean.

    Parameters:
        predictions (dict): {user_id: [top_k_movie_ids]}
        positives (dict): {user_id: set of relevant movie_ids}
        k (int): cutoff rank

    Returns:
        float: mean hit rate@k across all users (0 to 1)
    """
    hits = []
    for user_id, pred_movies in predictions.items():
        if user_id not in positives:
            continue
        relevant = positives[user_id]
        top_k = set(pred_movies[:k])
        hit = 1 if len(top_k & relevant) > 0 else 0
        hits.append(hit)

    return np.mean(hits) if hits else 0.0


def mrr_at_k(predictions, positives, k):
    """
    Mean Reciprocal Rank@K: mean of (1 / rank_of_first_positive) within top-K.

    For each user:
        - Find rank (1-indexed) of first positive item in top-K predictions.
        - If no positive in top-K, contribute 0.
        - MRR = mean of these reciprocals.

    Penalizes ranking errors: an item at rank 2 contributes 0.5, at rank 10 contributes 0.1.

    Parameters:
        predictions (dict): {user_id: [top_k_movie_ids]}
        positives (dict): {user_id: set of relevant movie_ids}
        k (int): cutoff rank

    Returns:
        float: mean reciprocal rank@k across all users (0 to 1)
    """
    mrrs = []
    for user_id, pred_movies in predictions.items():
        if user_id not in positives:
            continue
        relevant = positives[user_id]
        top_k = pred_movies[:k]

        # Find rank of first relevant item
        rank = None
        for i, movie_id in enumerate(top_k):
            if movie_id in relevant:
                rank = i + 1  # 1-indexed
                break

        mrr = (1.0 / rank) if rank is not None else 0.0
        mrrs.append(mrr)

    return np.mean(mrrs) if mrrs else 0.0


def evaluate(
    model_checkpoint_path,
    test_csv_path,
    metadata_json_path,
    train_csv_path=None,
    device=None,
    k_values=(10, 20),
    embed_dim=64,
    hidden_dim=None,
    dropout=0.0,
):
    """
    Full evaluation pipeline.

    Loads trained model, computes per-user item rankings on test set,
    and reports Recall@K, Hit Rate@K, MRR@K.

    Parameters:
        model_checkpoint_path (str): path to model_final.pt
        test_csv_path (str): path to test.csv (index-encoded)
        metadata_json_path (str): path to metadata.json
        train_csv_path (str or None): path to train.csv (needed to exclude train positives)
        device (torch.device or None): CPU or CUDA
        k_values (tuple): cutoffs to evaluate (e.g., (10, 20))
        embed_dim (int): embedding dimension (must match training)
        hidden_dim (int or None): hidden dimension in projection (must match training)
        dropout (float): dropout rate (must match training)

    Returns:
        dict: {
            'recall': {10: ..., 20: ...},
            'hit_rate': {10: ..., 20: ...},
            'mrr': {10: ..., 20: ...}
        }
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load metadata
    with open(metadata_json_path, "r") as f:
        metadata = json.load(f)
    num_users = metadata["num_users"]
    num_movies = metadata["num_movies"]
    logger.info(f"Loaded metadata: {num_users} users, {num_movies} movies")

    # Load test set
    test_df = pd.read_csv(test_csv_path)
    logger.info(f"Loaded test data: {len(test_df)} interactions")

    # Build positives dict: {user_id: set of relevant movie_ids}
    # For temporal test set with 1 positive per user, this is simple:
    positives = {}
    for _, row in test_df.iterrows():
        user_id = int(row['user_id'])
        movie_id = int(row['movie_id'])
        if user_id not in positives:
            positives[user_id] = set()
        positives[user_id].add(movie_id)
    logger.info(f"Built positives dict: {len(positives)} users with test items")

    # Load model
    model = TwoTowerModel(
        num_users=num_users,
        num_movies=num_movies,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )
    model.load_state_dict(
        torch.load(model_checkpoint_path, map_location=device, weights_only=True)
    )
    model = model.to(device)
    model.eval()
    logger.info(f"Loaded model from {model_checkpoint_path}")

    # Compute all user embeddings
    logger.info("Computing user embeddings...")
    user_ids_tensor = torch.arange(num_users, device=device)
    with torch.no_grad():
        user_embeddings = model.get_user_embeddings(user_ids_tensor)  # (num_users, embed_dim)
    logger.info(f"User embeddings shape: {user_embeddings.shape}")

    # Compute all movie embeddings
    logger.info("Computing movie embeddings...")
    movie_ids_tensor = torch.arange(num_movies, device=device)
    with torch.no_grad():
        movie_embeddings = model.get_item_embeddings(movie_ids_tensor)  # (num_movies, embed_dim)
    logger.info(f"Movie embeddings shape: {movie_embeddings.shape}")

    # Score all (user, movie) pairs via dot product
    logger.info("Scoring all (user, movie) pairs...")
    with torch.no_grad():
        scores = torch.matmul(user_embeddings, movie_embeddings.t())  # (num_users, num_movies)
    logger.info(f"Scores matrix shape: {scores.shape}")

    # Do not reward recommendations the user has already consumed.
    if train_csv_path:
        train_df = pd.read_csv(train_csv_path, usecols=["user_id", "movie_id"])
        user_idx = torch.as_tensor(train_df["user_id"].to_numpy(), device=device)
        movie_idx = torch.as_tensor(train_df["movie_id"].to_numpy(), device=device)
        scores[user_idx, movie_idx] = float("-inf")
        logger.info("Masked %d training interactions", len(train_df))

    # Get top-K predictions for each user
    if not k_values or any(k < 1 for k in k_values):
        raise ValueError("k_values must contain positive integers")
    max_k = min(max(k_values), num_movies)
    logger.info(f"Extracting top-{max_k} predictions per user...")
    top_k_scores, top_k_indices = torch.topk(scores, k=max_k, dim=1)
    # top_k_scores: (num_users, max_k)
    # top_k_indices: (num_users, max_k) — movie indices

    # Convert to predictions dict: {user_id: [movie_ids in ranked order]}
    predictions = {}
    for user_id in range(num_users):
        movie_ids = top_k_indices[user_id].cpu().numpy().tolist()
        predictions[user_id] = movie_ids

    # Evaluate at each k
    results = {
        'recall': {},
        'hit_rate': {},
        'mrr': {}
    }

    for k in k_values:
        recall = recall_at_k(predictions, positives, k)
        hit_rate = hit_rate_at_k(predictions, positives, k)
        mrr = mrr_at_k(predictions, positives, k)

        results['recall'][k] = recall
        results['hit_rate'][k] = hit_rate
        results['mrr'][k] = mrr

        logger.info(f"\n--- Metrics @ K={k} ---")
        logger.info(f"Recall@{k}:   {recall:.6f}")
        logger.info(f"Hit Rate@{k}: {hit_rate:.6f}")
        logger.info(f"MRR@{k}:      {mrr:.6f}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Two-Tower recommender")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/model_final.pt",
        help="Path to model checkpoint (default: checkpoints/model_final.pt)"
    )
    parser.add_argument(
        "--test_csv",
        type=str,
        default="data/processed/test.csv",
        help="Path to test CSV (default: data/processed/test.csv)"
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default="data/processed/metadata.json",
        help="Path to metadata JSON (default: data/processed/metadata.json)"
    )
    parser.add_argument(
        "--train_csv",
        type=str,
        default="data/processed/train.csv",
        help="Path to train CSV for excluding train positives (optional)"
    )
    parser.add_argument(
        "--embed_dim",
        type=int,
        default=64,
        help="Embedding dimension (must match training, default: 64)"
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=None,
        help="Hidden dimension in projection (must match training, default: same as embed_dim)"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Dropout rate (must match training, default: 0.0)"
    )

    args = parser.parse_args()

    results = evaluate(
        model_checkpoint_path=args.checkpoint,
        test_csv_path=args.test_csv,
        metadata_json_path=args.metadata,
        train_csv_path=args.train_csv,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )

    print("\n" + "="*50)
    print("FINAL EVALUATION RESULTS")
    print("="*50)
    print(json.dumps(results, indent=2))

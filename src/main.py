"""
main.py

FastAPI backend for Two-Tower recommendations.

Endpoints:
    POST /recommend — returns top-N movie IDs for a given user
    GET /health — health check

Design Decisions:

1. FastAPI (not Flask):
   - FastAPI is async-native, type-hinted, and auto-generates OpenAPI docs.
   - Better performance under concurrent requests (async IO).
   - Pydantic validation on request/response bodies.

2. Startup events to load model and index:
   - Model and FAISS index loaded once at server startup (not per request).
   - Embedded as lifespan context manager (FastAPI 0.93+).
   - Inference latency: ~5-10ms per user (FAISS retrieval only, no model forward).

3. User embedding cached on first request:
   - When a user is queried, compute their embedding and cache it.
   - Subsequent queries for the same user reuse the cached embedding.
   - Avoids redundant model forward passes.

4. No Postgres/database:
   - Movie metadata (titles, genres) loaded from metadata.json at startup.
   - For this portfolio project, simple in-memory is fine.
   - In production (step 6), could migrate to Postgres for scalability.

5. Minimal request validation:
   - User ID must exist in range [0, num_users).
   - K must be positive and ≤ num_movies.
   - Returns error JSON with HTTP 400 for invalid requests.
"""

import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Conditional import—FAISS CPU fallback if GPU not available
try:
    import faiss
except ImportError:
    raise ImportError("FAISS not installed. Run: pip install faiss-cpu")

try:  # Support `uvicorn src.main:app` from the repository root.
    from .model import TwoTowerModel
    from .inference import load_faiss_index
except ImportError:
    from model import TwoTowerModel
    from inference import load_faiss_index

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# =======================
# Global State (loaded at startup)
# =======================

model = None
faiss_index = None
user_embeddings_cache = {}
metadata = None
num_users = None
num_movies = None
device = None


# =======================
# Request/Response Models
# =======================

class RecommendRequest(BaseModel):
    user_id: int = Field(..., ge=0, description="User ID (0-indexed)")
    k: int = Field(default=10, ge=1, le=100, description="Number of recommendations")


class MovieRecommendation(BaseModel):
    movie_id: int
    rank: int
    score: float  # Inner product similarity (higher = more similar)


class RecommendResponse(BaseModel):
    user_id: int
    recommendations: list[MovieRecommendation]
    num_movies_in_corpus: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    index_loaded: bool
    num_users: Optional[int]
    num_movies: Optional[int]


# =======================
# Startup / Shutdown
# =======================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager: startup on enter, cleanup on exit.
    
    Loads model and FAISS index once when the server starts.
    """
    global model, faiss_index, metadata, num_users, num_movies, device

    logger.info("FastAPI startup: loading model and index...")

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load metadata
    metadata_path = os.getenv("METADATA_PATH", "data/processed/metadata.json")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    num_users = metadata["num_users"]
    num_movies = metadata["num_movies"]
    logger.info(f"Metadata loaded: {num_users} users, {num_movies} movies")

    # Load model
    checkpoint_path = os.getenv("MODEL_CHECKPOINT", "checkpoints/model_final.pt")
    model = TwoTowerModel(num_users=num_users, num_movies=num_movies)
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    model = model.to(device)
    model.eval()
    logger.info(f"Model loaded from {checkpoint_path}")

    # Load FAISS index
    index_path = os.getenv("FAISS_INDEX", "checkpoints/movie_embeddings.faiss")
    if os.path.exists(index_path):
        faiss_index = load_faiss_index(index_path, use_gpu=False)
        logger.info(f"FAISS index loaded from {index_path}")
    else:
        logger.warning(f"FAISS index not found at {index_path}. Run inference.py to build it.")

    logger.info("✓ Startup complete")

    yield  # Server runs here

    logger.info("FastAPI shutdown: cleaning up...")
    model = None
    faiss_index = None
    user_embeddings_cache.clear()
    logger.info("✓ Shutdown complete")


# =======================
# FastAPI App
# =======================

app = FastAPI(
    title="Two-Tower Recommender API",
    description="Fast approximate nearest-neighbor recommendations",
    version="1.0.0",
    lifespan=lifespan,
)


# =======================
# Endpoints
# =======================

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="ok" if model is not None and faiss_index is not None else "error",
        model_loaded=model is not None,
        index_loaded=faiss_index is not None,
        num_users=num_users,
        num_movies=num_movies,
    )


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(request: RecommendRequest):
    """
    Get top-K movie recommendations for a user.

    Parameters:
        user_id: User ID (0-indexed, must be < num_users)
        k: Number of recommendations (1-100)

    Returns:
        List of (movie_id, rank, score) tuples sorted by score (highest first).
    """
    global user_embeddings_cache, model, faiss_index

    if model is None or faiss_index is None:
        raise HTTPException(status_code=503, detail="Model or index not loaded")

    user_id = request.user_id
    k = min(request.k, num_movies)  # Cap k at corpus size

    # Validate user ID
    if user_id < 0 or user_id >= num_users:
        raise HTTPException(
            status_code=400,
            detail=f"user_id must be in range [0, {num_users})"
        )

    # Retrieve or compute user embedding
    if user_id not in user_embeddings_cache:
        logger.debug(f"Computing embedding for user {user_id}")
        with torch.no_grad():
            user_id_tensor = torch.tensor([user_id], device=device)
            user_emb = model.get_user_embeddings(user_id_tensor)
            user_emb = user_emb.cpu().numpy().astype(np.float32)
        user_embeddings_cache[user_id] = user_emb
    else:
        logger.debug(f"Using cached embedding for user {user_id}")
        user_emb = user_embeddings_cache[user_id]

    # FAISS retrieval using inner product similarity
    logger.debug(f"Retrieving top-{k} for user {user_id}")
    scores, indices = faiss_index.search(user_emb, k)

    # Parse results
    scores = scores[0]  # (k,)
    indices = indices[0]      # (k,)

    recommendations = [
        MovieRecommendation(
            movie_id=int(idx),
            rank=rank + 1,
            score=float(score),
        )
        for rank, (idx, score) in enumerate(zip(indices, scores))
    ]

    return RecommendResponse(
        user_id=user_id,
        recommendations=recommendations,
        num_movies_in_corpus=num_movies,
    )


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "service": "Two-Tower Recommender API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health (GET)",
            "recommend": "/recommend (POST)",
            "docs": "/docs (OpenAPI)",
        },
    }


if __name__ == "__main__":
    # For local testing: python -m uvicorn main:app --reload
    import uvicorn

    logger.info("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)

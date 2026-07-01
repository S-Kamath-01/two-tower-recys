"""
streamlit_app.py

Streamlit frontend for Two-Tower recommendations.
Connects to FastAPI backend for inference.
"""

import json
import logging
import os

import requests
import streamlit as st
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")
TIMEOUT = 10  # seconds


# =======================
# Cached Loading Functions
# =======================

@st.cache_resource
def load_metadata():
    """Load metadata with caching."""
    metadata_path = "data/processed/metadata.json"
    with open(metadata_path, "r") as f:
        return json.load(f)


@st.cache_resource
def load_movie_metadata():
    """Map index-encoded movie IDs to display metadata."""
    movies_dat = "data/raw/ml-1m/movies.dat"
    mapping_path = "data/processed/movie_to_idx.json"
    if not os.path.exists(movies_dat) or not os.path.exists(mapping_path):
        return {}

    try:
        with open(mapping_path, "r", encoding="utf-8") as f:
            movie_to_idx = {int(key): value for key, value in json.load(f).items()}
        movie_dict = {}
        with open(movies_dat, "r", encoding="latin-1") as f:
            for line in f:
                parts = line.rstrip("\n").split("::")
                original_id = int(parts[0])
                if len(parts) >= 2 and original_id in movie_to_idx:
                    movie_dict[movie_to_idx[original_id]] = {"title": parts[1]}
        return movie_dict
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        logger.warning("Failed to load movie metadata: %s", exc)
        return {}


def call_recommend_api(user_id: int, k: int) -> dict:
    """
    Call FastAPI /recommend endpoint.
    
    Returns:
        {"success": bool, "data": list | str}
        - On success: {"success": True, "data": [{"movie_id": int, "rank": int, "score": float}, ...]}
        - On error: {"success": False, "data": error_message}
    """
    try:
        response = requests.post(
            f"{API_URL}/recommend",
            json={"user_id": user_id, "k": k},
            timeout=TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        return {"success": True, "data": data.get("recommendations", [])}
    except requests.exceptions.ConnectionError:
        return {"success": False, "data": f"Cannot connect to API at {API_URL}. Is the backend running?"}
    except requests.exceptions.Timeout:
        return {"success": False, "data": f"API request timed out (>{TIMEOUT}s)"}
    except requests.exceptions.HTTPError as e:
        try:
            error_detail = e.response.json().get("detail", str(e))
        except:
            error_detail = str(e)
        return {"success": False, "data": f"API Error: {error_detail}"}
    except Exception as e:
        return {"success": False, "data": f"Unexpected error: {str(e)}"}


@st.cache_resource
def check_backend_health():
    """Check if backend is running."""
    try:
        response = requests.get(
            f"{API_URL}/health",
            timeout=5,
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}


# =======================
# Main Streamlit App
# =======================

def main():
    st.set_page_config(
        page_title="Two-Tower Recommender",
        page_icon="🎬",
        layout="wide"
    )

    st.title("🎬 Two-Tower Movie Recommender")
    st.markdown("""
    Enter a user ID to get personalized movie recommendations.
    
    **How it works:**
    - Neural two-tower architecture trained on MovieLens 1M data
    - Exact inner product search for nearest neighbors
    - Returns top-K ranked recommendations in milliseconds
    """)

    # Check backend health
    health_status = check_backend_health()
    if "error" in health_status:
        st.error(f"❌ Backend is not available: {health_status['error']}")
        st.info(f"Make sure the FastAPI backend is running at {API_URL}")
        st.stop()
    else:
        with st.sidebar:
            st.success("✅ Backend connected")

    # Load metadata
    try:
        metadata = load_metadata()
        movie_metadata = load_movie_metadata()
    except Exception as e:
        st.error(f"Failed to load metadata: {e}")
        st.stop()

    num_users = metadata.get("num_users", 0)
    num_movies = metadata.get("num_movies", 0)

    if num_users == 0 or num_movies == 0:
        st.error("Invalid metadata: num_users or num_movies is 0")
        st.stop()

    # Sidebar controls
    st.sidebar.header("⚙️ Settings")
    user_id = st.sidebar.number_input(
        "User ID",
        min_value=0,
        max_value=num_users - 1,
        value=0,
        step=1,
    )
    k = st.sidebar.slider(
        "Number of Recommendations (K)",
        min_value=1,
        max_value=min(50, num_movies),
        value=10,
        step=1,
    )

    # Get recommendations
    result = call_recommend_api(user_id, k)

    if not result["success"]:
        st.error(f"Error: {result['data']}")
        st.stop()

    recommendations = result["data"]

    if not recommendations:
        st.warning(f"No recommendations found for user {user_id}")
        st.stop()

    # Display results
    st.subheader(f"Top {k} Recommendations for User {user_id}")

    # Create results DataFrame
    results_data = []
    for rec in recommendations:
        rank = rec.get("rank", 0)
        movie_id = rec.get("movie_id", 0)
        score = rec.get("score", 0.0)

        movie_title = "N/A"
        if movie_id in movie_metadata:
            movie_title = movie_metadata[movie_id].get("title", "N/A")

        results_data.append({
            "Rank": rank,
            "Movie ID": movie_id,
            "Title": movie_title,
            "Score": f"{score:.6f}",
        })

    results_df = pd.DataFrame(results_data)
    st.dataframe(results_df, use_container_width=True, hide_index=True)

    # Additional info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Users", num_users)
    with col2:
        st.metric("Total Movies", num_movies)
    with col3:
        health_info = check_backend_health()
        model_loaded = health_info.get("model_loaded", False)
        index_loaded = health_info.get("index_loaded", False)
        st.metric("Backend Status", "Ready" if (model_loaded and index_loaded) else "Busy")

    st.divider()
    st.markdown("""
    ### About
    - **Architecture:** Two-Tower neural recommender with projection networks
    - **Loss:** Bayesian Personalized Ranking (BPR)
    - **Inference:** Exact inner product search
    - **Dataset:** MovieLens 1M with temporal leave-one-out split
    """)


if __name__ == "__main__":
    main()

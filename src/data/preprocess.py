import pandas as pd
import os
import pickle
import json
import argparse

def load_ratings(filepath):
    """
    Load the MovieLens ratings dataset.

    Parameters:
        filepath (str): Path to ratings.dat.

    Returns:
        pd.DataFrame:
            DataFrame containing:
            user_id, movie_id, rating, timestamp.
    """
    df = pd.read_csv(filepath, sep = "::", engine='python', names=["user_id","movie_id", "rating", "timestamp"])
    return df

def filter_positive(df, threshold=4):
    """
    Convert explicit ratings into implicit feedback.

    Keeps only interactions with rating >= threshold.

    Parameters:
        df (pd.DataFrame): Ratings DataFrame.
        threshold (int): Minimum rating considered positive.

    Returns:
        pd.DataFrame:
            Filtered DataFrame containing only
            positive interactions.
    """
    df = df[df['rating'] >= threshold]
    return df.copy()

def temporal_split(df):
    """
    Perform a leave-one-out temporal split.

    For each user:
        - Train set contains all interactions
          except the most recent one.
        - Test set contains the most recent
          interaction.

    Parameters:
        df (pd.DataFrame): Positive interactions.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            train, test
    """
    df = df.sort_values(["user_id", "timestamp"])
    test = df.groupby("user_id").tail(1)
    train = df.drop(test.index)
    return train, test

def build_id_mappings(positives):
    """
    Create contiguous integer mappings for users
    and movies.

    Original MovieLens IDs are converted into
    zero-based indices suitable for embedding
    layers.

    Parameters:
        positives (pd.DataFrame):
            Positive interaction DataFrame.

    Returns:
        tuple[dict, dict]:
            user_to_idx, movie_to_idx
    """
    unique_users = positives["user_id"].unique()
    user_to_idx = {user_id: idx for idx, user_id in enumerate(unique_users)}
    unique_movies = positives["movie_id"].unique()
    movie_to_idx = {movie_id: idx for idx, movie_id in enumerate(unique_movies)}
    return user_to_idx, movie_to_idx

def apply_mappings(df, user_to_idx, movie_to_idx):
    """
    Replace original MovieLens IDs with
    contiguous integer indices.

    Parameters:
        df (pd.DataFrame):
            Interaction DataFrame.
        user_to_idx (dict):
            User ID mapping.
        movie_to_idx (dict):
            Movie ID mapping.

    Returns:
        pd.DataFrame:
            DataFrame with encoded user_id
            and movie_id columns.
    """
    df = df.copy()
    df["user_id"] = df["user_id"].map(user_to_idx)
    df["movie_id"] = df["movie_id"].map(movie_to_idx)
    return df


def save_processed(train, test, user_to_idx, movie_to_idx, threshold, output_dir="../data/processed"):
    """
    Save processed datasets and metadata.

    Artifacts saved:
        - train.csv
        - test.csv
        - user_to_idx.pkl
        - movie_to_idx.pkl
        - metadata.json

    Parameters:
        train (pd.DataFrame):
            Training interactions.
        test (pd.DataFrame):
            Test interactions.
        user_to_idx (dict):
            User mapping dictionary.
        movie_to_idx (dict):
            Movie mapping dictionary.
        threshold (int):
            Positive rating threshold.
        output_dir (str):
            Directory where artifacts are saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    train.to_csv(os.path.join(output_dir, "train.csv"), index=False)

    test.to_csv(os.path.join(output_dir, "test.csv"), index=False)

    with open(os.path.join(output_dir, "user_to_idx.pkl"), "wb") as f:
        pickle.dump(user_to_idx, f)

    with open(os.path.join(output_dir, "movie_to_idx.pkl"), "wb") as f:
        pickle.dump(movie_to_idx, f)

    # JSON copy is safe to load in serving/UI processes. Pickle files should
    # only be opened when their provenance is trusted.
    with open(os.path.join(output_dir, "movie_to_idx.json"), "w") as f:
        json.dump({str(key): value for key, value in movie_to_idx.items()}, f)

    metadata = {
    "threshold": threshold,
    "num_users": len(user_to_idx),
    "num_movies": len(movie_to_idx),
    "num_train_interactions": len(train),
    "num_test_interactions": len(test)
    }

    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f)


def main():
    parser = argparse.ArgumentParser(description="Preprocess MovieLens ratings")
    parser.add_argument("--ratings", default="data/raw/ml-1m/ratings.dat")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--threshold", type=float, default=4.0)
    args = parser.parse_args()

    ratings = load_ratings(args.ratings)
    positives = filter_positive(ratings, args.threshold)
    if positives.empty:
        raise ValueError("No positive interactions remain after filtering")
    user_to_idx, movie_to_idx = build_id_mappings(positives)
    train, test = temporal_split(positives)
    train = apply_mappings(train, user_to_idx, movie_to_idx)
    test = apply_mappings(test, user_to_idx, movie_to_idx)
    save_processed(
        train, test, user_to_idx, movie_to_idx, args.threshold, args.output_dir
    )
    print(
        f"Saved {len(train):,} train and {len(test):,} test interactions "
        f"to {args.output_dir}"
    )


if __name__ == "__main__":
    main()

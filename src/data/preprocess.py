import pandas as pd
import os
import pickle
import json

def load_ratings(filepath):
    df = pd.read_csv(filepath, sep = "::", engine='python', names=["user_id","movie_id", "rating", "timestamp"])
    return df

def filter_positive(df, threshold=4):
    df = df[df['rating'] >= threshold]
    return df

def temporal_split(df):
    df = df.sort_values(["user_id", "timestamp"])
    test = df.groupby("user_id").tail(1)
    train = df.drop(test.index)
    return train, test

def build_id_mappings(positives):
    unique_users = positives["user_id"].unique()
    user_to_idx = {user_id: idx for idx, user_id in enumerate(unique_users)}
    unique_movies = positives["movie_id"].unique()
    movie_to_idx = {movie_id: idx for idx, movie_id in enumerate(unique_movies)}
    return user_to_idx, movie_to_idx

def apply_mappings(df, user_to_idx, movie_to_idx):
    df = df.copy()
    df["user_id"] = df["user_id"].map(user_to_idx)
    df["movie_id"] = df["movie_id"].map(movie_to_idx)
    return df


def save_processed(train, test, user_to_idx, movie_to_idx, threshold, output_dir="../data/processed"):
    train.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    test.to_csv(os.path.join(output_dir, "test.csv"), index=False)

    with open(os.path.join(output_dir, "user_to_idx.pkl"), "wb") as f:
        pickle.dump(user_to_idx, f)

    with open(os.path.join(output_dir, "movie_to_idx.pkl"), "wb") as f:
        pickle.dump(movie_to_idx, f)

    metadata = {
        "threshold": threshold,
        "num_users": len(user_to_idx),
        "num_movies": len(movie_to_idx)
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f)

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class TwoTowerDataset(Dataset):
    """
    PyTorch Dataset for BPR-style pairwise training of a
    two-tower recommender.

    Each item returned is a triplet:
        (user_id, positive_movie_id, negative_movie_id)

    One epoch = one pass over all positive interactions in
    train.csv, each paired with a freshly sampled negative.

    Parameters:
        train_path (str): Path to train.csv (already
            user/movie-index-encoded).
        num_movies (int): Total number of distinct movies,
            used as the sampling range for negatives.
    """

    def __init__(self, train_df, num_movies):
        self.interactions = train_df[['user_id', 'movie_id']].values
        self.user_positives = (train_df.groupby('user_id')['movie_id'].apply(set).to_dict())
        self.num_movies = num_movies


    def __len__(self):
        return len(self.interactions)

    def _sample_negative(self, user_id):
        """
        Sample a single negative movie_id for a given user via
        rejection sampling: draw uniformly from all movies until
        one is found that the user did not positively interact
        with in train.
        Parameters:
            user_id (int): Index-encoded user_id
            
        
        Returns:
            int: A movie_id not in the user's positive set.
        """

        positives = self.user_positives[user_id]
        while True:
            candidate = np.random.randint(0, self.num_movies)
            if candidate not in positives:
                return candidate

    def __getitem__(self, idx):
        """
        Return one BPR training triplet.
        
        Parameters:
            idx (int): Index into self.interactions.
            
        Returns:
            tuple[int , int, int]:
                (user_id, positive_movie_id, negative_movie_id)
                
        """

        user_id, pos_movie_id = self.interactions[idx]
        user_id = int(user_id)
        pos_movie_id = int(pos_movie_id)
        neg_movie_id = self._sample_negative(user_id)
        return user_id, pos_movie_id, neg_movie_id
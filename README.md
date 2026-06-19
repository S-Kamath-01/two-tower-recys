# 🎬 Two-Tower Recommender System

A retrieval-based movie recommendation system built using a Two-Tower neural architecture and trained on implicit feedback from the MovieLens 1M dataset.

The project follows a modern recommendation pipeline consisting of:

* Data preprocessing
* Temporal train/test splitting
* Pairwise retrieval training
* Embedding-based candidate generation
* Ranking evaluation
* Model serving via FastAPI
* Interactive recommendation interface via Streamlit

Recommendations are generated using dot-product similarity between learned user and movie embeddings.

---

## Overview

Modern recommendation systems are typically implemented as multi-stage retrieval and ranking pipelines.

This project focuses on the retrieval stage by learning separate user and movie representations using a Two-Tower architecture. User and movie embeddings are projected into a shared latent space, where relevance is measured through dot-product similarity.

---

## Architecture

```text
                 User ID
                    │
                    ▼
          ┌─────────────────┐
          │ User Embedding  │
          └─────────────────┘
                    │
                    ▼
               User Vector

                 Movie ID
                    │
                    ▼
          ┌─────────────────┐
          │ Item Embedding  │
          └─────────────────┘
                    │
                    ▼
               Movie Vector

      Score = User Vector · Movie Vector
```

---

## Tech Stack

| Component       | Technology               |
| --------------- | ------------------------ |
| Language        | Python                   |
| ML Framework    | PyTorch                  |
| Data Processing | Pandas, NumPy            |
| Database        | PostgreSQL               |
| Backend         | FastAPI                  |
| Frontend        | Streamlit                |
| Deployment      | Render + Streamlit Cloud |
| Version Control | Git + GitHub             |

---

## Dataset

Dataset: MovieLens 1M

### Original Statistics

| Metric       | Value     |
| ------------ | --------- |
| Users        | 6,040     |
| Movies       | 3,706     |
| Interactions | 1,000,209 |
| Sparsity     | 95.53%    |

### Implicit Feedback Conversion

Explicit ratings are converted into implicit feedback:

```python
rating >= 4
```

Ratings below 4 are discarded.

Resulting positive interactions:

```text
575,281
```

---

## Preprocessing Pipeline

Implemented in:

```text
src/data/preprocess.py
```

Pipeline:

```text
Raw Ratings
      │
      ▼
Filter Positive Interactions
      │
      ▼
Temporal Leave-One-Out Split
      │
      ▼
User & Movie ID Encoding
      │
      ▼
Processed Artifacts
```

### Temporal Evaluation Strategy

For every user:

```text
Train → all interactions except latest
Test  → latest interaction
```

This prevents temporal leakage and better reflects real-world recommendation scenarios.

---

## Dataset Construction

Implemented in:

```text
src/data/dataset.py
```

Training uses a custom PyTorch Dataset for pairwise recommendation learning.

### Training Format

Each training example is represented as:

```text
(user_id, positive_movie_id, negative_movie_id)
```

Example:

```text
(12, 83, 742)
```

Meaning:

* User 12 positively interacted with Movie 83
* Movie 742 is sampled as a negative example

### Dynamic Negative Sampling

Negative samples are generated during training using rejection sampling.

For a user u:

```text
negative_movie ∉ positive_history(u)
```

User histories are built exclusively from training interactions to avoid information leakage from held-out test data.

---

## Dataset Statistics

### Post-Filtering Statistics

| Metric                                       | Value   |
| -------------------------------------------- | ------- |
| Users Retained                               | 6,038   |
| Movies Retained                              | 3,533   |
| Positive Interactions                        | 575,281 |
| Users With Exactly One Positive Interaction  | 1       |
| Movies With Exactly One Positive Interaction | 152     |

### Training Statistics

| Metric                   | Value   |
| ------------------------ | ------- |
| Train Interactions       | 569,243 |
| Test Interactions        | 6,038   |
| Users With Train History | 6,037   |
| Movies                   | 3,533   |

---

## Methodology

### Implicit Feedback

Only ratings greater than or equal to 4 are treated as positive interactions.

### Temporal Evaluation

Future interactions are never used during training.

### Pairwise Learning

Training examples are represented as:

```text
(user, positive_item, negative_item)
```

rather than binary classification labels.

### Dynamic Negative Sampling

Negative samples are generated during training instead of being precomputed.

### Two-Tower Retrieval

User and movie embeddings are learned independently and compared using dot-product similarity.

---

## Evaluation

The retrieval model will be evaluated using ranking-based metrics:

* Recall@K
* Hit Rate@K
* Mean Reciprocal Rank (MRR)

Metrics such as RMSE and Accuracy are intentionally not used because recommendation is fundamentally a retrieval and ranking problem.

---

## Project Structure

```text
two-tower-recsys/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocess_testing.ipynb
│   └── 03_dataset_testing.ipynb
├── src/
│   ├── data/
│   │   ├── preprocess.py
│   │   └── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
├── models/
├── streamlit_app/
├── tests/
├── README.md
├── LICENSE
└── requirements.txt
```

---

## Roadmap

### Completed

* Dataset exploration
* Implicit feedback conversion
* Temporal train/test split
* User/movie ID encoding
* Processed artifact generation
* Pairwise dataset construction
* Dynamic negative sampling

### In Progress

* Two-Tower retrieval model

### Planned

* BPR training pipeline
* Ranking evaluation
* FastAPI serving layer
* Streamlit frontend
* Dockerization
* Cloud deployment

---

## Installation

```bash
git clone https://github.com/<your-username>/two-tower-recsys.git
cd two-tower-recsys

pip install -r requirements.txt
```

---

## Contributing

Contributions, suggestions, issue reports, and code reviews are welcome.

For significant changes, please open an issue before submitting a pull request.

---

## License

This project is licensed under the MIT License.

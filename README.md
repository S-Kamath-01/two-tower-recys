# 🎬 Two-Tower Recommender System

A production-style movie recommendation system built on the MovieLens 1M dataset using a **Two-Tower Neural Retrieval Architecture**.

The goal of this project is to understand modern recommendation systems from first principles while building a complete end-to-end ML system, including data preprocessing, model training, evaluation, serving, and deployment.

---

## 🚀 Project Goals

* Build an interview-defensible recommendation system
* Learn embedding-based retrieval architectures
* Understand negative sampling and ranking losses
* Implement an end-to-end ML pipeline
* Deploy a recommendation service using FastAPI and Streamlit

---

## 🏗️ System Architecture

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

The model learns separate embeddings for users and movies. Recommendations are generated using dot-product similarity between learned embeddings.

---

## 🛠️ Tech Stack

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

## 📊 Dataset

**Dataset:** MovieLens 1M

### Original Dataset Statistics

| Metric       | Value     |
| ------------ | --------- |
| Users        | 6,040     |
| Movies       | 3,706     |
| Interactions | 1,000,209 |
| Sparsity     | 95.53%    |

### Implicit Feedback Conversion

The original MovieLens dataset contains explicit ratings from 1–5.

For retrieval training, ratings are converted to implicit feedback:

```python
rating >= 4
```

Resulting positive interactions:

```text
575,281
```

---

## ⚙️ Preprocessing Pipeline

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
Processed Training Artifacts
```

### Train/Test Split Strategy

A temporal leave-one-out split is used:

* Train → all interactions except the latest interaction
* Test → latest interaction

This prevents information leakage that would occur with random train/test splits.

---

## 📈 Post-Filtering Statistics

| Metric                                       | Value   |
| -------------------------------------------- | ------- |
| Users Retained                               | 6,038   |
| Movies Retained                              | 3,533   |
| Positive Interactions                        | 575,281 |
| Users With Exactly One Positive Interaction  | 1       |
| Movies With Exactly One Positive Interaction | 152     |

### Key Findings

* User-item interaction matrix is **95.53% sparse**
* Strong long-tail popularity distribution exists
* 173 movies have no positive interactions after filtering
* Recommendation is treated as a **ranking problem**, not a rating prediction problem

---

## 📂 Project Structure

```text
two-tower-recsys/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_preprocess_testing.ipynb
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
└── requirements.txt
```

---

## ✅ Progress Tracker

### Data Pipeline

* [x] Environment setup
* [x] MovieLens dataset acquisition
* [x] Exploratory data analysis
* [x] Sparsity analysis
* [x] Implicit feedback conversion
* [x] Temporal train/test split
* [x] User/movie ID encoding
* [x] Processed artifact generation

### Model Development

* [ ] PyTorch Dataset implementation
* [ ] Negative sampling
* [ ] Two-tower architecture
* [ ] Training pipeline
* [ ] Ranking evaluation

### Serving & Deployment

* [ ] PostgreSQL integration
* [ ] FastAPI backend
* [ ] Streamlit frontend
* [ ] Cloud deployment

---

## 📏 Evaluation Metrics

The recommender will be evaluated using ranking-based metrics:

* Precision@K
* Recall@K

Metrics such as RMSE and Accuracy are intentionally not used because recommendation is fundamentally a retrieval and ranking task.

---

## 🔮 Future Improvements

Potential V2 enhancements:

* Hard negative mining
* FAISS-based ANN retrieval
* Cold-start strategies
* Hybrid recommendation features
* Real-time recommendation logging
* Retrieval + ranking pipeline

---

## 📚 Learning Objectives

This project is intentionally being built from scratch to deeply understand:

* Embeddings
* Collaborative filtering
* Negative sampling
* Retrieval systems
* Recommendation system evaluation
* Production ML pipelines

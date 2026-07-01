# Two-Tower Movie Recommender

[![CI](https://github.com/S-Kamath-01/two-tower-recys/actions/workflows/ci.yml/badge.svg)](https://github.com/S-Kamath-01/two-tower-recys/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

An end-to-end recommendation system built with PyTorch, FAISS, FastAPI, and
Streamlit. It learns separate user and movie representations from implicit
MovieLens 1M feedback and retrieves recommendations using exact inner-product
search.

## What is included

- Temporal leave-one-out preprocessing for implicit feedback
- Dynamic negative sampling and Bayesian Personalized Ranking (BPR) loss
- Two-tower PyTorch model with independently computable item embeddings
- Recall@K, Hit Rate@K, and MRR@K evaluation
- FAISS `IndexFlatIP` retrieval
- FastAPI inference service and Streamlit interface
- Docker Compose configuration and GitHub Actions tests

## Architecture

```text
User ID  -> User tower  -> User embedding  --+
                                               +-> dot product -> score
Movie ID -> Movie tower -> Movie embedding --+
```

Because the towers are independent, movie embeddings can be computed once and
stored in a retrieval index. This repository uses exact FAISS search because
MovieLens 1M has only a few thousand movies.

## Repository layout

```text
src/
  data/preprocess.py   Dataset preprocessing CLI
  data/dataset.py      Pairwise training dataset
  model.py             Two-tower model
  train.py             BPR training CLI
  evaluate.py          Offline evaluation CLI
  inference.py         FAISS index builder
  main.py              FastAPI service
streamlit_app/app.py   Web interface
tests/                 Unit tests
notebooks/             Exploration and pipeline walkthroughs
```

Data, virtual environments, `.env`, model weights, and generated indexes are
intentionally excluded from Git. A fresh clone therefore needs the preparation
steps below before the application can start.

## Setup

Requirements: Python 3.10 or newer and approximately 4 GB of available RAM.

```bash
git clone https://github.com/S-Kamath-01/two-tower-recys.git
cd two-tower-recys
python -m venv venv
```

Activate the environment:

```powershell
# Windows PowerShell
.\venv\Scripts\Activate.ps1
```

```bash
# macOS or Linux
source venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Prepare data and artifacts

1. Download `ml-1m.zip` from the
   [MovieLens 1M dataset page](https://grouplens.org/datasets/movielens/1m/).
2. Extract it so that `data/raw/ml-1m/ratings.dat` exists.
3. Run the pipeline from the repository root:

```bash
python -m src.data.preprocess
python -m src.train --epochs 50
python -m src.evaluate
python -m src.inference
```

Training writes `checkpoints/model_final.pt`; index building writes
`checkpoints/movie_embeddings.faiss`. These generated files are deliberately
not committed.

For a quick smoke test, use fewer epochs:

```bash
python -m src.train --epochs 1 --output_dir checkpoints/smoke
```

## Run locally

Start the API from the repository root:

```bash
python -m uvicorn src.main:app --reload --port 8000
```

In a second terminal with the same environment activated:

```bash
python -m streamlit run streamlit_app/app.py
```

Open:

- Web interface: <http://localhost:8501>
- API documentation: <http://localhost:8000/docs>
- Health check: <http://localhost:8000/health>

Example request:

```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": 0, "k": 10}'
```

## Docker Compose

Docker also expects the processed data, trained checkpoint, and FAISS index to
exist locally first.

```bash
docker compose up --build
```

This starts the API on port `8000` and Streamlit on port `8501`.

## Tests

```bash
python -m unittest discover -s tests -v
python -m compileall -q src streamlit_app
```

The same checks run automatically through GitHub Actions.

## API

### `GET /health`

Reports whether metadata, the model, and the FAISS index loaded successfully.

### `POST /recommend`

```json
{
  "user_id": 0,
  "k": 10
}
```

The response contains ranked index-encoded movie IDs and their inner-product
scores. These IDs are internal model indices, not original MovieLens IDs.

## Evaluation notes

The preprocessing step keeps each user's latest positive interaction for the
test set. Evaluation masks training interactions before ranking, preventing
already-consumed movies from inflating the reported metrics. Results depend on
the random initialization and training configuration; this README intentionally
does not claim benchmark numbers that are not produced by CI.

## Privacy and data handling

- MovieLens data and all derived interaction files remain local and ignored.
- `.env` is ignored; only the non-secret `.env.example` template is published.
- Checkpoints and FAISS indexes are ignored because they are generated artifacts.
- The API accepts an index-encoded user ID only and has no authentication. Do not
  expose it publicly without adding access control, rate limiting, and HTTPS.

MovieLens data is distributed under the terms stated by GroupLens. Review those
terms before redistributing the dataset or using it commercially.

## Limitations

- ID-only towers cannot recommend for unseen users or movies.
- One sampled negative per positive is simple but not necessarily optimal.
- The service does not yet filter previously consumed movies at online serving
  time; the masking is currently part of offline evaluation only.
- In-memory user embedding caching is process-local and unbounded.

## References

- [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/)
- [BPR: Bayesian Personalized Ranking from Implicit Feedback](https://arxiv.org/abs/1205.2618)
- [FAISS](https://github.com/facebookresearch/faiss)

## License

The source code is available under the [MIT License](LICENSE). The MovieLens
dataset has separate terms and is not covered by this repository's MIT license.

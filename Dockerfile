# Use official Python runtime as base image
# Python 3.12 for PyTorch compatibility
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY streamlit_app/ ./streamlit_app/
COPY data/processed/ ./data/processed/
COPY checkpoints/ ./checkpoints/

# Set environment variables (can be overridden at runtime)
ENV PYTHONUNBUFFERED=1
ENV MODEL_CHECKPOINT=/app/checkpoints/model_final.pt
ENV FAISS_INDEX=/app/checkpoints/movie_embeddings.faiss
ENV METADATA_PATH=/app/data/processed/metadata.json

# Expose port
EXPOSE 8000

# Change to src directory for proper imports (main.py uses bare imports like 'from model import')
WORKDIR /app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=5)"

# Run FastAPI app with Uvicorn
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]

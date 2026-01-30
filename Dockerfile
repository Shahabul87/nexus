FROM python:3.11-slim

WORKDIR /app

# System dependencies for audio/image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ src/
COPY api/ api/
COPY models/ models/

# Set Python path
ENV PYTHONPATH=/app/src:/app

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

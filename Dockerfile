# HuggingFace Spaces Docker SDK — NEXUS Streamlit Demo
# Docs: https://huggingface.co/docs/hub/spaces-sdks-docker
# Build: 2026-02-11

FROM python:3.12-slim

# Create non-root user (required by HF Spaces)
RUN useradd -m -u 1000 user
ENV PATH="/home/user/.local/bin:$PATH"

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install as user
COPY --chown=user ./requirements_spaces.txt requirements_spaces.txt
RUN pip install --no-cache-dir --upgrade -r requirements_spaces.txt

# Switch to non-root user
USER user

# Copy source code
COPY --chown=user ./src/ src/
COPY --chown=user ./models/ models/
COPY --chown=user ./app.py .

# Set environment
ENV PYTHONPATH=/app/src
ENV STREAMLIT_SERVER_PORT=7860
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 7860

CMD ["python", "-m", "streamlit", "run", "src/demo/streamlit_app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]

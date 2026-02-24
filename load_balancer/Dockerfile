FROM python:3.9-slim

# Set working directory to the app
WORKDIR /app

# Install docker-compose and LightGBM runtime dependency (libgomp)
RUN apt-get update && \
    apt-get install -y docker-compose libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Copy telemetry proto package (needed for gRPC server)
# Build context must be the project root (see docker-compose.yml)
COPY telemetry/ ./telemetry/

# Install requirements
COPY load_balancer/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy load balancer source code
COPY load_balancer/*.py .

# Copy precomputed feature mask (fallback if Redis is unavailable)
COPY load_balancer/models/feature_mask.json /app/models/
ENV MASK_FALLBACK_PATH=/app/models/feature_mask.json

# Run the fastAPI when the container starts
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

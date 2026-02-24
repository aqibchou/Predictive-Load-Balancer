FROM python:3.9-slim

# Set working directory to the app
WORKDIR /app

# Install requirements
# Build context must be the project root (see docker-compose.yml)
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend server code and telemetry agent
COPY backend/server.py .
COPY backend/telemetry_agent.py .

# Copy telemetry proto package (generated pb2 stubs)
COPY telemetry/ ./telemetry/

# Run the backend FastAPI when container starts
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]

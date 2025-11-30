FROM python:3.9-slim

# Set working directory to the app
WORKDIR /app

# Install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend server code
COPY server.py .

# Run the backend FastAPI when container starts
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
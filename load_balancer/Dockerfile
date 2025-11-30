FROM python:3.9-slim 

# Set working directory to the app
WORKDIR /app

# Install requirements 
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy server code from machine -> to this container 
COPY *.py .

# Run the fastAPI when the container starts 
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
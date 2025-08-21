FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Locust files
COPY . .

# Make scripts executable
RUN chmod +x distributed_runner.py

# Expose Locust web UI port
EXPOSE 8089

# Set environment variables
ENV PYTHONPATH=/app
ENV LOCUST_HOST=http://localhost:8080

# Default command (can be overridden)
CMD ["locust", "--web-host=0.0.0.0", "--web-port=8089"]
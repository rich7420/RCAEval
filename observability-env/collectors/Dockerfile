FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directory
RUN mkdir -p /app/data

# Set environment variables
ENV PYTHONPATH=/app
ENV PROMETHEUS_URL=http://prometheus:9090
ENV LOKI_URL=http://loki:3100
ENV JAEGER_URL=http://jaeger:16686

# Expose port for health checks
EXPOSE 8000

# Default command (can be overridden)
CMD ["python", "-m", "http.server", "8000"]
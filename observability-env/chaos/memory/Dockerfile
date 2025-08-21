FROM ubuntu:22.04

# Install stress-ng and Python dependencies
RUN apt-get update && apt-get install -y \
    stress-ng \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install psutil

# Copy stress injection script
COPY memory_stress.py /usr/local/bin/memory_stress.py
RUN chmod +x /usr/local/bin/memory_stress.py

# Create working directory
WORKDIR /chaos

# Default command
CMD ["python3", "/usr/local/bin/memory_stress.py", "--help"]
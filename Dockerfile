# Use Python slim image as a base
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install required system packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git && \
    rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Install additional dependencies
RUN pip install git+https://github.com/huggingface/transformers && \
    pip install librosa && \
    pip install python-dotenv pymilvus[model]

# Expose port (if Flask or any other app needs to be exposed)
EXPOSE 5000

# Run the Python app
CMD ["python", "api.py"]

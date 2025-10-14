# syntax=docker/dockerfile:1.6

# Use a small base image
FROM python:3.10-slim

# Prevent Python from writing .pyc files and forcing stdout buffering
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies for common Python libs (optional but useful)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency file first for better caching
COPY requirements.txt .

# Use BuildKit’s persistent cache for pip
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the rest of the project
COPY . .

# Expose the port (Cloud Run defaults to 8080)
EXPOSE 8080

# Run FastAPI with Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]

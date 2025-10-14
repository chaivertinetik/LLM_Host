# syntax=docker/dockerfile:1.6

FROM python:3.10-slim

# Speed + cleaner logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Only keep this if you actually need to compile wheels or use git+https deps.
# If not needed, delete this whole RUN to make builds much faster/smaller.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
 && rm -rf /var/lib/apt/lists/*

# Copy deps first for layer caching
COPY requirements.txt .

# Use BuildKit cache for pip wheels
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install -r requirements.txt

# Now copy the app code
COPY . .

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]

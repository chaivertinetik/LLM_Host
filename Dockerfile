# syntax=docker/dockerfile:1.6

FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Optional: only keep if you compile wheels or use git+https dependencies
# RUN apt-get update && apt-get install -y --no-install-recommends \
#    build-essential git \
# && rm -rf /var/lib/apt/lists/*

# Layer-cached deps
COPY requirements.txt .

# BuildKit cache for pip wheels (works because buildx uses BuildKit)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install -r requirements.txt

# App code
COPY . .

EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]

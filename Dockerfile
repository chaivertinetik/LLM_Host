# Use an official Python image
FROM python:3.10

# Set the working directory
WORKDIR /app

# Copy all project files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Cloud Run port (optional but doesn't hurt)
EXPOSE 8080

# Start FastAPI server with Uvicorn
CMD ["sh", "-c", "uvicorn llmagent:app --host 0.0.0.0 --port ${PORT:-8080}"]

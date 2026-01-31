# Use a lightweight Python base image
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Install system dependencies (ffmpeg is required for Whisper audio processing)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code into the container
COPY ./src ./src
COPY ./data ./data

# Expose the port the app will run on
EXPOSE 8000

# Command to run the app (using FastAPI/Uvicorn for a robust backend)
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
# ===== Dockerfile (CPU version) =====
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libpoppler-cpp-dev \
    ffmpeg \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy all project files
COPY . /app

# Install dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu


# Run your main script when the container starts
CMD ["python", "src/data_processing.py"]

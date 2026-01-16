FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpoppler-cpp-dev \
    ffmpeg \
    libgl1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Cache-friendly dependency installation
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Copy source after dependencies
COPY src ./src
COPY data ./data

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]

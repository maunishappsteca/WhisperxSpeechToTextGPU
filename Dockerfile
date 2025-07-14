# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies required by librosa & soundfile
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy Python files
COPY requirements.txt .
COPY app.py .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Add this before your pip install:
RUN pip install huggingface-hub


# Add model download step:
RUN python -c "\
import os; \
from huggingface_hub import snapshot_download; \
model_size = os.getenv('WHISPER_MODEL', 'large-v3'); \
cache_dir = os.getenv('WHISPER_MODEL_CACHE', '/app/models'); \
print(f'Downloading {model_size} to {cache_dir}'); \
snapshot_download(repo_id=f'openai/whisper-{model_size}', \
                  local_dir=os.path.join(cache_dir, model_size), \
                  local_dir_use_symlinks=False, \
                  resume_download=True)"


# Run the app
CMD ["python", "app.py"]


FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Don't buffer Python output (important for logs in Docker/Railway)
ENV PYTHONUNBUFFERED=1

# Default: run the full system (Slack bot + scheduler)
CMD ["python", "main.py"]

# Use Python 3.11 slim image
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy project files
COPY . .

# Ensure model exists (train if not present)
RUN python train.py || echo "Skipping training"

# Expose correct port for Render
EXPOSE 10000

# Health check (correct URL)
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:10000/docs')" || exit 1

# Run API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "10000"]
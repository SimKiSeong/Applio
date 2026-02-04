# syntax=docker/dockerfile:1
FROM python:3.10-bullseye

EXPOSE 6969
WORKDIR /app

# System deps
RUN apt-get update && \
    apt-get install -y -qq --no-install-recommends \
      ffmpeg \
      libportaudio2 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy app
COPY . .

# Install Python deps (NO venv inside Docker)
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    python -m pip install --no-cache-dir python-ffmpeg && \
    python -m pip install --no-cache-dir \
      torch==2.7.1 \
      torchvision==0.22.1 \
      torchaudio==2.7.1 \
      --index-url https://download.pytorch.org/whl/cu128 && \
    if [ -f "requirements.txt" ]; then \
      python -m pip install --no-cache-dir -r requirements.txt; \
    fi

VOLUME ["/app/logs/"]

ENTRYPOINT ["python"]
CMD ["app.py"]

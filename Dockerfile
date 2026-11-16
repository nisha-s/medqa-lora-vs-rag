# CPU-only image for local debugging. Not meant for training runs.
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# CPU wheels for torch keep the image small.
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

COPY . .

ENV PYTHONPATH=/app/src

CMD ["bash", "scripts/run_all.sh", "configs/debug.yaml"]

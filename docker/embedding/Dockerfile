FROM python:3.10-slim

WORKDIR /app

ENV PYTHONPATH=/app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY docker/embedding/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

CMD ["python", "-m", "Data.embedding_extraction"]

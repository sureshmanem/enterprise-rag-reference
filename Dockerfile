FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 ANONYMIZED_TELEMETRY=False
WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

COPY data ./data
RUN useradd -m app && chown -R app /app
USER app

EXPOSE 8000
CMD ["sh", "-c", "rag ingest data/sample && rag serve --host 0.0.0.0 --port 8000"]

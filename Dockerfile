FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

RUN python -m pip install --no-cache-dir -r requirements.txt \
    && python -m pip check

RUN useradd --create-home --uid 10001 appuser

COPY --chown=appuser:appuser src/ ./src/

USER appuser

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "predictive_maintenance.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

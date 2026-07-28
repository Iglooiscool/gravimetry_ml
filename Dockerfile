FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md /app/
COPY src /app/src

RUN pip install --upgrade pip \
    && pip install -e .[dev] \
    && python -c "import tensorflow as tf; print(tf.__version__)"

CMD ["python", "-c", "import gravimetry_ml; print('Environment ready: gravimetry_ml')"]

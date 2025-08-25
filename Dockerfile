FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY static ./static

EXPOSE 7860
# Hugging Face Spaces fournit $PORT
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}
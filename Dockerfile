FROM python:3.13-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir gunicorn

COPY app.py ./
COPY templates ./templates
COPY artifacts ./artifacts

EXPOSE 5000

CMD ["sh", "-c", "gunicorn -b 0.0.0.0:${PORT:-5000} app:app"]
FROM python:3.10-slim

WORKDIR /app

# Biar log aplikasi langsung keluar ke stdout/stderr
ENV PYTHONUNBUFFERED=1 \
    PORT=8010

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8010

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c 'import os, urllib.request; urllib.request.urlopen(f"http://127.0.0.1:{os.getenv(\"PORT\", \"8010\")}/health", timeout=3)'

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]

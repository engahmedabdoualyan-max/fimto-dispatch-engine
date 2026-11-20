# ── Dispatch Optimizer (OR-Tools CP-SAT) ────────────────────────────────────
# OR-Tools ships manylinux wheels with native extensions, so we pin a modern
# Debian-based Python image (glibc) instead of Alpine/musl.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY samples ./samples

# OR-Tools is CPU bound: one worker process per container, scale horizontally.
# Cloud hosts (Render, Fly, Cloud Run) inject $PORT and route traffic to it;
# fall back to 8000 for local `docker run`.
ENV PORT=8000
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT','8000'))"

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]

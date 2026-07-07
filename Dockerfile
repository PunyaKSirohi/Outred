FROM python:3.12-slim

WORKDIR /app

# Create non-root user for security (R6)
RUN groupadd --system outred && useradd --system --gid outred outred

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Ensure the non-root user owns the app directory
RUN chown -R outred:outred /app

# Switch to non-root user
USER outred

EXPOSE 8000

# Health check (R7) — verifies the app is responding
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" || exit 1

CMD ["python", "run.py"]
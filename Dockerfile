# Dockerfile

# --- Stage 1: Builder ---
FROM python:3.12-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Stage 2: Final Production Image ---
FROM python:3.12-slim
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY ./src /app/src
# Also copy the data and mlruns directories needed by the simulation
COPY ./data /app/data
COPY ./mlruns /app/mlruns
EXPOSE 80
CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "80"]

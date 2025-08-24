# --- Stage 1: The "Builder" Stage ---
# Use a full-featured base image to install dependencies safely
FROM python:3.11-slim as builder

WORKDIR /app

# Install system dependencies that some python packages might need
RUN apt-get update && apt-get install -y --no-install-recommends build-essential

# Copy only the dependency file first to leverage Docker layer caching
COPY requirements.txt .

# Install python dependencies
RUN pip wheel --no-cache-dir --wheel-dir /app/wheels -r requirements.txt


# --- Stage 2: The "Final" Stage ---
# Start from a minimal base image for the final application
FROM python:3.11-slim

WORKDIR /app

# Copy the pre-built wheels from the builder stage
COPY --from=builder /app/wheels /wheels/

# Install the wheels without needing build tools
RUN pip install --no-cache /wheels/*

# Copy the application source code
COPY src/ ./src

# Expose the port the app runs on
EXPOSE 8000

# Command to run the Uvicorn server
# It will listen on all network interfaces (0.0.0.0) on port 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
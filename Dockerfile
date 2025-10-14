# ----------- Dockerfile -----------
FROM python:3.10-slim

# System dependencies (needed for numpy, pandas, lightgbm)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app/

EXPOSE 8080
CMD ["gunicorn", "-b", "0.0.0.0:${PORT}", "serve_api:app"]

FROM python:3.11.4-slim

# Install system libs needed by OpenCV (for libGL.so.1)
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      libgl1 \
      libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY req.txt .
RUN pip install --no-cache-dir -r req.txt

COPY . .

CMD ["python", "app.py"]

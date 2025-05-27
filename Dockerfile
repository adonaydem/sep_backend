# 1. Choose Python base image
FROM python:3.11.4-slim

# 2. Set working directory
WORKDIR /app

# 3. Copy requirements and install
COPY req.txt .
RUN pip install --no-cache-dir -r req.txt

# 4. Copy the rest of your code
COPY . .

# 5. Run the app
CMD ["python", "app.py"]

FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
ENV MLFLOW_TRACKING_URI="file:/app/mlruns_docker"
CMD ["python", "src/train.py"]

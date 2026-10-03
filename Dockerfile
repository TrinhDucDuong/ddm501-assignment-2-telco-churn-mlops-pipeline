FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip pip install --timeout 180 --retries 5 -r requirements.txt
COPY churn churn
COPY config.yaml .
COPY data data
CMD ["python", "-m", "churn"]

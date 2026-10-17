FROM python:3.10-slim
ENV PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt
COPY app/ ./app/
# models/model.pkl is produced by `python src/train.py` and must exist before build
COPY models/model.pkl ./models/model.pkl
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

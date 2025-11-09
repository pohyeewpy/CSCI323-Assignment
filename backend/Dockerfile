# ReviewLens backend
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1     PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends     build-essential curl &&     rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py ./
COPY reviews.csv ./

COPY best_model.joblib ./
COPY vectorizer.joblib ./ 

EXPOSE 8000
ENV CSV_PATH=/app/reviews.csv

RUN useradd -m appuser
USER appuser

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

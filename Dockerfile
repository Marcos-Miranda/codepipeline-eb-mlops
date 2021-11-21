FROM continuumio/miniconda3:4.8.3

COPY ./requirements.txt \
     ./src/fraud_detector/app.py \
     ./src/fraud_detector/prediction.py \
     /app/

COPY ./models /app/models

WORKDIR /app

RUN pip install --upgrade pip && \
    pip install --no-cache-dir --upgrade -r requirements.txt

CMD ["python", "app.py"]
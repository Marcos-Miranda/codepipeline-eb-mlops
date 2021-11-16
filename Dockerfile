FROM continuumio/miniconda3:4.8.3

COPY ./requirements.txt /app/requirements.txt
COPY ./models /models
COPY ./src/app.py /app/app.py
COPY ./src/prediction.py /app/prediction.py

WORKDIR /app

RUN pip install --upgrade pip && \
    pip install --no-cache-dir --upgrade -r requirements.txt

CMD ["python", "app.py"]
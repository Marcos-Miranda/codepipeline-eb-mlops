SHELL := /bin/bash

prepare:
	mkdir models
	mkdir data
	aws s3 cp s3://mlops1/dados_fraude.tsv ./data/dados_fraude.tsv

install:
	pip install --upgrade pip
	pip install -e .
	pip install -r requirements_dev.txt

lint:
	flake8 -v src
	mypy src --exclude app.py

test:
	pytest --maxfail=1 tests/test_feature_engineering.py
	pytest --maxfail=1 tests/test_model_training.py
	pytest --maxfail=1 tests/test_prediction.py

train:
	python -m fraud_detector.model_training

run-app:
	python -m fraud_detector.app

build-image:
	docker build -t model-serving .

run-docker:
	docker run -p 8080:8080 -it --rm model-serving
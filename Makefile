install:
	pip install --upgrade pip &&\
	pip install -r requirements.txt

lint:
	flake8 --max-line-length=119 -v src

test:
	if [ ! -f ./tests/dados_fraude.tsv]; then\
		aws s3 cp s3://mlops1/dados_fraude.tsv ./tests/dados_fraude.tsv;\
	fi
	pytest --maxfail=1

all: install test lint

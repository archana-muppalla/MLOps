setup:
	python3 -m venv ./MLOps
	source ./MLOps/bin/activate

install:
	pip install --upgrade pip && pip install -r requirements.txt
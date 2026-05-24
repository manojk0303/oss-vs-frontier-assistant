.PHONY: install ui cli eval charts test docker

install:
	pip install -r requirements.txt

ui:
	python -m app.gradio_app

cli-frontier:
	python -m app.cli --backend frontier

cli-oss:
	python -m app.cli --backend oss

eval:
	python -m evaluation.run_eval --backends oss,frontier

eval-frontier:
	python -m evaluation.run_eval --backends frontier

charts:
	python -m evaluation.make_charts

test:
	pytest -q tests/

docker:
	docker build -t ollive-assistant -f deployment/Dockerfile .

docker-run:
	docker run --rm -p 7860:7860 --env-file .env ollive-assistant

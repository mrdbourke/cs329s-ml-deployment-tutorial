.PHONY: run run-container gcloud-deploy

APP_NAME ?= food-vision

run:
	@streamlit run app.py --server.port=8080 --server.address=0.0.0.0

run-container:
	@docker build . -t ${APP_NAME}
	@docker run -p 8080:8080 ${APP_NAME}

gcloud-deploy:
	@gcloud app deploy app.yaml


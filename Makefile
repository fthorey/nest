SHELL := bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help
.DELETE_ON_ERROR:
.SUFFIXES:

DOCKERFILE = ./Dockerfile
TRITON_SERVER_VERSION = 21.02-py3

.PHONY: help
help: ## Hello friends!
	$(info Available make targets:)
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

.PHONY: build
build: ## Building docker image: cthorey/nest
	$(info *** Building docker image: cthorey/nest)
	@docker build \
    --platform linux/amd64 \
		--tag cthorey/nest \
		--file $(DOCKERFILE) .


.PHONY: install-udev-rules
install-udev-rules: ## Install udev rules
	$(info *** Install udev rule)
	@sudo cp scripts/10-camera.rules /etc/udev/rules.d/10-camera.rules &&\
	sudo udevadm control --reload-rules && sudo service udev restart && sudo udevadm trigger

.PHONY: deploy
deploy: ## Deploy YOLOV5 to torchscript -- needed to run before starting the app
	$(info *** Deploy the model)
	@docker run --rm -ti \
		--platform linux/amd64 \
    --volume $(HOME)/workdir/nest:/workdir \
		cthorey/nest ./scripts/deploy.py

.PHONY: start
start: ## Start the app
	$(info *** Start the app)
	@docker-compose up -d

.PHONY: shutdown
shutdown: ## Shutdown the app
	$(info *** Start the app)
	@docker-compose down

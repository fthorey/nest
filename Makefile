MAKEFLAGS += --warn-undefined-variables --no-print-directory
SHELL := bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := build
.DELETE_ON_ERROR:
.SUFFIXES:
ASSET_FOLDER = ~/workdir/Outboard/assets
DOCKERFILE = ./Dockerfile
TRITON_SERVER_VERSION = 21.02-py3

.PHONY: build
build:
	$(info *** Building docker image: cthorey/nest)
	@docker build \
		--tag cthorey/nest \
		--file $(DOCKERFILE) .

.PHONY: install-udev-rules
install-udev-rules:
	$(info *** Install udev rule)
	@sudo cp scripts/10-camera.rules /etc/udev/rules.d/10-camera.rules &&\
	sudo udevadm control --reload-rules && sudo service udev restart && sudo udevadm trigger

.PHONY: start
start:
	$(info *** Start the app)
	@docker-compose up -d

.PHONY: shutdown
shutdown:
	$(info *** Start the app)
	@docker-compose down

.PHONY: triton-server
triton-server: ## Launch a triton inference server. Then you can use their client and start making prediction
	$(info *** Launch a triton inference server)
	@docker run -d --rm \
    --shm-size=1g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -p 8000:8000 \
    -p 8001:8001 \
    -p 8002:8002 \
    --name trt_server_cont \
    --volume $(ASSET_FOLDER)/data/common/triton_models:/models \
    nvcr.io/nvidia/tritonserver:$(TRITON_SERVER_VERSION) tritonserver --model-store=/models --log-verbose=2

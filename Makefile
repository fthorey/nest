MAKEFLAGS += --warn-undefined-variables --no-print-directory
SHELL := bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := build
.DELETE_ON_ERROR:
.SUFFIXES:

DOCKERFILE = ./Dockerfile

.PHONY: build
build:
	$(info *** Building docker image: cthorey/nest)
	@docker build \
		--no-cache \
		--tag cthorey/nest \
		--file $(DOCKERFILE) .

.PHONY: install-udev-rules
install-udev-rules:
	$(info *** Install udev rule)
	@sudo cp scripts/10-camera.rules /etc/udev/rules.d/10-camera.rules &&\
	sudo udevadm control --reload-rules && sudo service udev restart && sudo udevadm trigger

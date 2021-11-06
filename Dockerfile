FROM python:3.9-slim-buster AS base

# change shell
RUN rm /bin/sh && ln -s /bin/bash /bin/sh

# Install some deps for outcloud opencv
RUN export TZ=Europe/Paris
RUN apt update && \
    apt install -y curl git build-essential ffmpeg libsm6 libxext6 pkg-config && \
    apt clean

# python
COPY requirements.txt /requirements.txt
RUN pip3 install -r /requirements.txt


ENV PYTHONPATH=$PYTHONPATH:/workdir
COPY ./scripts/jupyter_notebook_config.py /root/.jupyter/

WORKDIR /workdir
COPY . /workdir
RUN pip install -e .
RUN fc-cache -rv
ENV PASSWORD=hello
CMD ["bash"]

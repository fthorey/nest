FROM osrf/ros:noetic-desktop

RUN apt update && \
    apt install -y python3-pip git wget cmake curl build-essential libffi-dev libpq-dev && \
    apt clean

RUN apt update && \
    apt install -y ros-noetic-camera-info-manager ros-noetic-cv-camera && \
    apt clean

# change shell
RUN rm /bin/sh && ln -s /bin/bash /bin/sh

# python
COPY requirements.txt /requirements.txt
RUN pip3 install -r /requirements.txt
RUN pip install -qr https://raw.githubusercontent.com/ultralytics/yolov5/master/requirements.txt


ENV PYTHONPATH=$PYTHONPATH:/workdir
COPY jupyter_notebook_config.py /root/.jupyter/

# current librry
WORKDIR /workdir
COPY scripts /workdir/scripts
RUN mkdir /workdir/data

ENV PASSWORD=hello
COPY scripts/entrypoint.sh /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]

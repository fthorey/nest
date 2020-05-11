FROM osrf/ros:kinetic-desktop-full

# change shell
RUN rm /bin/sh && ln -s /bin/bash /bin/sh

# setup ros
RUN mkdir /catkin_ws &&\
  mkdir /catkin_ws/src &&\
  cd /catkin_ws/src &&\
  git clone https://github.com/OTL/cv_camera.git
RUN source /opt/ros/kinetic/setup.bash && \
  cd /catkin_ws && \
  catkin_make

# python
COPY requirements.txt /requirements.txt
RUN apt update &&\
  apt install -y python-pip
RUN python2 -m pip install --upgrade pip==9.0.3 && \
  python2 -m pip install --requirement /requirements.txt
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

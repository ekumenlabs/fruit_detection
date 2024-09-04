FROM ubuntu:22.04 AS training_prod

COPY training_ws /root/training_ws/

WORKDIR /root/training_ws/
RUN mkdir data
RUN mkdir out

RUN apt-get update
RUN apt-get install -y python3.10 python3-pip
RUN python3 -m pip install -r requirements.txt

ENTRYPOINT [ "/usr/bin/bash" ]
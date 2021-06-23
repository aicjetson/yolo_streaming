FROM nvcr.io/nvidia/l4t-ml:r32.5.0-py3

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install -U pip \
    && pip3 install uwsgi==2.0.19.1 \
    && pip3 install flask==1.1.2

RUN apt-get update \
    && apt-get install nginx -y \
    && apt-get install protobuf-compiler libprotoc-dev libjpeg-dev cmake -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN python3 --version \
    && pip3 --version

WORKDIR /usr/app

CMD ["bash"]

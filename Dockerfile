FROM harbor.sia-service.kr/sia/cuda:11.1-cudnn8-py3.8.5-pytorch1.9-220406
ENV PIP_INDEX_URL=https://pypi.sia-service.kr/simple/

# due to the issue https://github.com/NVIDIA/nvidia-docker/issues/1631
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list

# for gdal install
RUN apt-get update

RUN pip install mmcv-full==1.4.7 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html

WORKDIR /clasymm

COPY . /clasymm
RUN pip3 install --no-cache-dir -e ".[optional]"

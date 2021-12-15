FROM docker.siadev.kr/sia/cuda:11.1-cudnn8-py3.8.5-torch1.9

# for gdal install
RUN apt-get update
RUN apt-get install libgdal-dev -y

WORKDIR /libs
RUN curl -O http://download.osgeo.org/gdal/2.4.1/gdal-2.4.1.tar.gz
RUN tar -xzvf gdal-2.4.1.tar.gz
WORKDIR /libs/gdal-2.4.1/swig/python/
RUN python3 setup.py build && python3 setup.py install

COPY submodules/siavision/ /ardet/submodules/siavision
RUN cd /ardet/submodules/siavision \
    && pip3 install .

COPY submodules/dataset /ardet/submodules/dataset
RUN cd /ardet/submodules/dataset \
    && pip3 install .

RUN cd /libs \
    && git clone --depth 1 -b v1.3.16 https://github.com/open-mmlab/mmcv \
    && cd mmcv && MMCV_WITH_OPS=1 pip3 install .

RUN pip3 install mmcls==0.18.0

WORKDIR /clasymm

COPY . /clasymm
# RUN pip3 install .

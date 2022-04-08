FROM docker.siadev.kr/sia/cuda:11.1-cudnn8-py3.8.5-torch1.9
ENV PIP_INDEX_URL=https://pypi.sia-service.kr/simple/

# for gdal install
RUN apt-get update
RUN apt-get install libgdal-dev -y

WORKDIR /libs
RUN curl -O http://download.osgeo.org/gdal/2.4.1/gdal-2.4.1.tar.gz
RUN tar -xzvf gdal-2.4.1.tar.gz
WORKDIR /libs/gdal-2.4.1/swig/python/
RUN python3 setup.py build && python3 setup.py install

# Install SIALibs.dataset
COPY submodules/dataset /clasymm/submodules/dataset
RUN cd /clasymm/submodules/dataset \
    && pip3 install .

# Install GEOCOCO
COPY submodules/geococo/ /clasymm/submodules/geococo
RUN cd /clasymm/submodules/geococo/PythonAPI \
 && python3 setup.py build_ext install

RUN cd /libs \
    && git clone --depth 1 -b v1.3.16 https://github.com/open-mmlab/mmcv \
    && cd mmcv && MMCV_WITH_OPS=1 pip3 install .

WORKDIR /clasymm

COPY . /clasymm
RUN pip3 install --no-cache-dir -e ".[optional]"

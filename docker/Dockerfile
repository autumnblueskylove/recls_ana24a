FROM harbor.sia-service.kr/sia/cuda:11.1-cudnn8-py3.8.5-pytorch1.9-220406
ENV PIP_INDEX_URL=https://pypi.sia-service.kr/simple/


# due to the issue https://github.com/NVIDIA/nvidia-docker/issues/1631
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list

# for gdal install
RUN apt-get update

# install gdal
WORKDIR /libs
RUN curl -O http://download.osgeo.org/gdal/2.4.1/gdal-2.4.1.tar.gz
RUN tar -xzvf gdal-2.4.1.tar.gz
WORKDIR /libs/gdal-2.4.1/swig/python/
RUN python3 setup.py build && python3 setup.py install

# Set environment variables
ENV JP2KAK_THREADS=5
ENV GDAL_DATA=/libs/gdal-2.4.1/data

# Install MIM
RUN pip3 install openmim

# Install recls
WORKDIR /recls
COPY . /recls
RUN mim install --no-cache-dir -e ".[Optional]"

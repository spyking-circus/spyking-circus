FROM continuumio/miniconda3

# `pip install pyqt5` seems to consistently fail, so use a conda base and use conda to install pyqt...

#########################################
### Python, etc                                                                                                                
RUN apt-get update && apt-get -y install git wget build-essential
RUN apt-get install -y python3 python3-pip
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y python3-tk

RUN apt-get update && apt-get install -y libmpich-dev mpich \
    qt5-default \ 
    libglib2.0-0 libgl1-mesa-glx \ 
    packagekit-gtk3-module libcanberra-gtk-module libcanberra-gtk3-module

#########################################
### Parallel HDF
RUN echo "Installing Parallel HDF..."

ENV CC=mpicc
ENV HDF5_MPI="ON"
# where parallel hdf5 will be built
ENV HDF5_DIR="/root/hdf5-1.12.0/hdf5"

WORKDIR /root

RUN wget -O hdf5.tar.gz https://www.hdfgroup.org/package/hdf5-1-12-0-tar-gz/?wpdmdl=14582

RUN tar -xf hdf5.tar.gz

WORKDIR /root/hdf5-1.12.0

RUN ./configure --enable-parallel --enable-shared && \
    make && \
    make install

WORKDIR /root

RUN echo "Installing H5PY with MPI support"
RUN pip3 install --no-binary=h5py h5py

#########################################
### Spyking Circus
RUN echo "Installing SpyKING CIRCUS and phy 2.0..."

RUN pip3 install scikit-build
RUN pip3 install cmake
RUN pip3 install spyking-circus

### Phy
RUN pip3 install colorcet pyopengl qtconsole requests traitlets tqdm joblib click mkdocs dask toolz mtscomp
# this is why we use `continuumio/miniconda3` instead of ubuntu:18.04
RUN conda install pyqt cython pillow 
RUN pip3 install phy --pre --upgrade


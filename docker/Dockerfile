FROM ubuntu:18.04

#########################################
### Python, etc                                                                                                                
RUN apt-get update && apt-get -y install git wget build-essential
RUN apt-get install -y python3 python3-pip
RUN ln -s python3 /usr/bin/python
RUN ln -s pip3 /usr/bin/pip
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y python3-tk

RUN echo "Installing SpyKING CIRCUS and phy 2.0..."

#########################################
### Spyking Circus
RUN apt-get update && apt-get install -y libmpich-dev mpich
RUN pip install spyking-circus

### Phy
RUN pip install pyqt5 colorcet pyopengl qtconsole requests traitlets tqdm joblib click mkdocs
RUN pip install https://github.com/cortex-lab/phylib/archive/master.zip
RUN pip install https://github.com/cortex-lab/phy/archive/dev.zip
RUN apt-get update && apt-get install -y libglib2.0-0
RUN apt-get update && apt-get install -y libgl1-mesa-glx
RUN apt-get update && apt-get install -y qt5-default
RUN apt-get update && apt-get install -y packagekit-gtk3-module
RUN apt-get update && apt-get install -y libcanberra-gtk-module libcanberra-gtk3-module
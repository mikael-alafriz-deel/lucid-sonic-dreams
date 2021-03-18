FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

RUN apt-get update &&\
    apt-get -y install build-essential \
    cmake unzip git wget tmux nano curl \
    sysstat libtcmalloc-minimal4 pkgconf autoconf libtool \
    python3 python3-pip python3-dev python3-setuptools \
    libsm6 libxext6 libxrender-dev \
    libhdf5-100 libhdf5-dev \
    libasound-dev libportaudio2 libsndfile1 \
    ffmpeg &&\
    ln -s /usr/bin/python3 /usr/bin/python &&\
    ln -s /usr/bin/pip3 /usr/bin/pip &&\
    apt-get clean &&\
    apt-get autoremove &&\
    rm -rf /var/lib/apt/lists/* &&\
    rm -rf /var/cache/apt/archives/*

# Install pip and setuptools
RUN pip3 install --upgrade --no-cache-dir \
    pip==21.0.1 \
    setuptools==53.0.0 \
    packaging==20.9

# Install python packages
COPY requirements.txt /requirements.txt
RUN pip3 install --no-cache-dir -r /requirements.txt
# Install lucid-sonic-dreams from source
COPY . /lucid-sonic-dreams
RUN cd lucid-sonic-dreams &&\
    python setup.py install

# Fix jupyter https://github.com/jupyter/jupyter_console/issues/163
RUN pip install --upgrade ipykernel

WORKDIR /workdir
CMD /lucid-sonic-dreams/run_colab_notebook.sh

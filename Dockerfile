FROM nvcr.io/nvidia/pytorch:21.05-py3

# Avoid interaction with apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Update the repository and install necessary packages (curl, wget etc.)
RUN apt-get update && apt-get -y -qq install --no-install-recommends \
    apt-utils \
    build-essential \
    curl \
    wget \
    git \
    nano \
    openssh-client \
    screen

# Copy requirements.txt to the docker image
COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install -q -r requirements.txt
RUN pip install torch==1.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
RUN pip install torchvision==0.11.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
RUN pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.1+cu113.html
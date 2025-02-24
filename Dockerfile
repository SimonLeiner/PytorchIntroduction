FROM python:3.12-slim

# set working directory
WORKDIR /workspace

# Update packages, install git, and clean up
RUN apt-get update && apt-get install --no-install-recommends  -y \
    # github
    git \
    # for installing packages
    curl \
    # for interacting with remote servers
    openssh-client\
    # clean up
    && apt-get clean\
    && rm -rf /var/lib/apt/lists/*

# copy and install requirments
COPY requirements.txt requirements.txt
RUN pip install -U -r requirements.txt
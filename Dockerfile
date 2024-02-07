FROM python:3.11-slim

# Update packages, install git, curl, and openssh-client and clean up
RUN apt-get update && apt-get install --no-install-recommends  -y \
    git \
    curl \
    gcc  python3-dev \
    openssh-client\
    && apt-get clean\
    && rm -rf /var/lib/apt/lists/* 

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
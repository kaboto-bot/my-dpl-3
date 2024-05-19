FROM python:3.7-slim

USER root

RUN apt-get update && \
    apt-get -y install --reinstall ca-certificates && \
    apt-get -y install software-properties-common && \
    pip install --upgrade pip
    
# Install Basic Packages
RUN apt-get install -y gcc python3-dev
# RUN pip install ipykernel jupyter
RUN pip install numpy==1.16
# Azure ML does not support Debian 10 yet, so cannot use Anaconda image as base
FROM tensorflow/tensorflow:1.15.2-py3

COPY ./requirements.txt .
RUN apt update & apt upgrade -y
RUN pip install -r requirements.txt
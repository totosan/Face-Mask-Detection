# Azure ML does not support Debian 10 yet, so cannot use Anaconda image as base
FROM tensorflow/tensorflow:1.15.2-py3

COPY ./requirements.txt .
RUN apt update -y && apt install git -y
RUN /usr/bin/python3 -m pip install --upgrade pip
RUN pip install -r requirements.txt
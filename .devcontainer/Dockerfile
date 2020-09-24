# Azure ML does not support Debian 10 yet, so cannot use Anaconda image as base
FROM tensorflow/tensorflow:1.15.2-py3

RUN apt update & apt upgrade & apt install git
RUN pip install -r ../requirements.txt
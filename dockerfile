FROM python:3.8-slim-buster

COPY ./requirements.txt /app/requirements.txt
WORKDIR /app
RUN apt update
RUN pip3 install -r requirements.txt
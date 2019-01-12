FROM ubuntu:16.04

COPY . .


RUN apt-get update
RUN apt-get -y upgrade

RUN apt-get install -y tmux python3 python-setuptools python-dev python-pip


RUN pip install -r requirements.txt


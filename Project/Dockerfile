FROM tensorflow/tensorflow:latest-gpu
WORKDIR /App

COPY requirements.txt requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY src src
COPY Config Config
COPY Main.py Main.py
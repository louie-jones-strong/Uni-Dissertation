FROM tensorflow/tensorflow:2.12.0-gpu
WORKDIR /app

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install --upgrade pip


COPY ./src/Learner/requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY main.py main.py

COPY .vscode .vscode
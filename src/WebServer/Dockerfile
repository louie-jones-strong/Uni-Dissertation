FROM python:3.8.10
WORKDIR /app



RUN pip install --upgrade pip

COPY src/WebServer/requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY main.py main.py

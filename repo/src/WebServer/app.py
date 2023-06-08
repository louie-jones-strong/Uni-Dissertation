from flask import Flask
from WebServer.views import views

app = Flask(__name__)
app.register_blueprint(views, url_prefix="/")

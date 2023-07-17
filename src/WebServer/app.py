from flask import Flask
from src.WebServer.views import views


def Run() -> None:
	app = Flask(__name__)
	app.register_blueprint(views, url_prefix="/")
	app.run(debug=True, host="0.0.0.0", port=5000)

	return
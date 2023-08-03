from flask import Flask
from src.WebServer.views import views
import os


def CleanUpReplays() -> None:
	replaysFolder = os.path.join("src", "WebServer", "static", "replays")
	if os.path.exists(replaysFolder):
		for file in os.listdir(replaysFolder):
			os.remove(os.path.join(replaysFolder, file))
	else:
		os.makedirs(replaysFolder)

	return

def Run() -> None:

	CleanUpReplays()

	app = Flask(__name__)
	app.register_blueprint(views, url_prefix="/")
	app.run(debug=True, host="0.0.0.0", port=5000)

	return
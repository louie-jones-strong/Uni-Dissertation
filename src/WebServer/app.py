from flask import Flask
from src.WebServer.views import views
import os


def CleanUpAssets() -> None:
	assetsFolder = os.path.join("src", "WebServer", "static", "assets")
	if os.path.exists(assetsFolder):
		for file in os.listdir(assetsFolder):
			os.remove(os.path.join(assetsFolder, file))
	else:
		os.makedirs(assetsFolder)

	return

def Run() -> None:

	CleanUpAssets()

	app = Flask(__name__)
	app.register_blueprint(views, url_prefix="/")
	app.run(debug=True, host="0.0.0.0", port=5000)

	return
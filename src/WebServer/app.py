from flask import Flask
import src.WebServer.views as views
import src.WebServer.AssetCreator as AssetCreator

class WebServer:

	def __init__(self, envConfig) -> None:
		AssetCreator.CleanUpAssets()
		viewsBlueprint = views.Setup(envConfig)

		self.App = Flask(__name__)
		self.App.register_blueprint(viewsBlueprint, url_prefix="/")

		return

	def Run(self) -> None:
		self.App.run(debug=True, host="0.0.0.0", port=5000)
		return
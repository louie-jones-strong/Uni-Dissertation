from flask import Blueprint, render_template, redirect
import typing
import os
import src.Common.Utils.PathHelper as PathHelper
import src.Common.EpisodeReplay.EpisodeReplay as ER
import cv2 as cv
import src.WebServer.AssetCreator as AssetCreator




def Setup(envConfig) -> None:

	rootPath = PathHelper.GetRootPath()
	DataFolder = os.path.join(rootPath, "Data", envConfig["Name"])
	RunFolder = os.path.join(DataFolder, envConfig["Group"])
	ReplaysFolder = os.path.join(RunFolder, "replays")

	reverbClient = None
	try:
		import reverb
		reverbClient = reverb.Client(f"experience-store:{5001}")
	except:
		print("Reverb not installed")

	views = Blueprint("views", __name__)






	def GetCommonData() -> typing.Dict[str, typing.Any]:
		data = {}

		if not os.path.exists(ReplaysFolder):
			os.makedirs(ReplaysFolder)

		replays = os.listdir(ReplaysFolder)

		data["EpisodeReplays"] = replays
		data["envConfig"] = envConfig

		actionLabels = {}
		for key, value in envConfig["AgentConfig"]["Human"]["Controls"].items():
			actionLabels[value] = key
		data["ActionLabels"] = actionLabels


		return data







#region endpoints
	@views.route("/")
	def Home() -> str:
		data = GetCommonData()
		return render_template("overview.html", **data)


	@views.route("/metrics")
	def Metrics() -> str:
		# redirect to wandb page
		return redirect("https://wandb.ai/louiej-s/Dissertation", code=302)


	@views.route("/episodereview/<episode>")
	def EpisodeReview(episode) -> str:
		data = GetCommonData()

		if episode not in data["EpisodeReplays"]:
			return "Episode not found"

		replayPath = os.path.join(ReplaysFolder, episode)
		replay = ER.EpisodeReplay.LoadFromFolder(replayPath)

		data["replayData"] = replay

		# make video for replay
		AssetCreator.CreateVideo(replay)

		return render_template("episodereview.html", **data)


	@views.route("/episodereview/<episode>/action/<action>")
	def ReviewAction(episode, action) -> str:
		action = int(action)

		data = GetCommonData()

		if episode not in data["EpisodeReplays"]:
			return "Episode not found"

		replayPath = os.path.join(ReplaysFolder, episode)
		replay = ER.EpisodeReplay.LoadFromFolder(replayPath)

		action = min(action, len(replay.Steps) - 1)
		action = max(action, 0)

		data["replayData"] = replay
		data["actionData"] = replay.Steps[action]
		data["actionIndex"] = action

		# make video for replay
		AssetCreator.CreateImage(replay, action)

		return render_template("actionreview.html", **data)
#endregion endpoints

	@views.route("/api/saveTrajectories", methods=["POST"])
	def SaveTrajectories() -> str:
		if reverbClient is None:
			return "Reverb not installed"
		else:
			checkPointPath = reverbClient.checkpoint()
			print(checkPointPath)
			return checkPointPath

		return

	return views
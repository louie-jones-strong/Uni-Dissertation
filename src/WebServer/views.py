from flask import Blueprint, render_template, redirect
import typing
import os
import src.Common.Utils.PathHelper as PathHelper
import src.Common.EpisodeReplay.EpisodeReplay as ER
import cv2 as cv
import src.WebServer.AssetCreator as AssetCreator

views = Blueprint("views", __name__)


rootPath = PathHelper.GetRootPath()
DataFolder = os.path.join(rootPath, "Data", "FrozenLake")
RunFolder = os.path.join(DataFolder, "dev")
ReplaysFolder = os.path.join(RunFolder, "replays")



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

	replays = os.listdir(ReplaysFolder)
	if episode not in replays:
		return "Episode not found"

	firstReplay = os.path.join(ReplaysFolder, episode)
	replay = ER.EpisodeReplay.LoadFromFolder(firstReplay)

	data["replayData"] = replay

	# make video for replay
	AssetCreator.CreateVideo(replay)



	return render_template("episodereview.html", **data)



def GetCommonData() -> typing.Dict[str, typing.Any]:
	data = {}

	replays = os.listdir(ReplaysFolder)

	data["EpisodeReplays"] = replays
	data["Environment"] = "FrozenLake"


	return data
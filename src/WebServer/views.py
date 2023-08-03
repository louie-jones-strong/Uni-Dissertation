from flask import Blueprint, render_template, redirect
import typing
import os
import src.Common.Utils.PathHelper as PathHelper
import src.Common.EpisodeReplay.EpisodeReplay as ER
import cv2 as cv

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


@views.route("/episodereview")
def EpisodeReview() -> str:
	data = GetCommonData()

	replays = os.listdir(ReplaysFolder)
	firstReplay = os.path.join(ReplaysFolder, replays[0])
	replay = ER.EpisodeReplay.LoadFromFolder(firstReplay)

	data["replayData"] = replay

	# make video for replay
	CreateVideo(replay)



	return render_template("episodereview.html", **data)





def CreateVideo(replay:ER.EpisodeReplay) -> None:

	firstFrame = replay.Steps[0].HumanState

	if firstFrame is None:
		return

	width = firstFrame.shape[1]
	height = firstFrame.shape[0]
	frameDups = 5
	outputPath = os.path.join("src", "WebServer", "static", "assets", f"{replay.EpisodeId}.mp4")


	# create the video
	fourcc = cv.VideoWriter_fourcc(*"H264")
	videoWriter = cv.VideoWriter(outputPath, fourcc, 25, (width, height))


	for step in replay.Steps:
		for i in range(frameDups):
			# numpy array to image
			image = cv.cvtColor(step.HumanState, cv.COLOR_RGB2BGR)
			videoWriter.write(image)

	videoWriter.release()
	return


def GetCommonData() -> typing.Dict[str, typing.Any]:
	data = {}


	return data
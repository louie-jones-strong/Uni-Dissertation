import os
import src.Common.EpisodeReplay.EpisodeReplay as ER
import cv2 as cv

def CreateVideo(replay:ER.EpisodeReplay) -> None:

	firstFrame = replay.Steps[0].HumanState

	if firstFrame is None:
		return

	width = firstFrame.shape[1]
	height = firstFrame.shape[0]
	frameDups = 25
	outputPath = os.path.join("src", "WebServer", "static", "assets", f"{replay.EpisodeId}.mp4")


	# create the video
	fourcc = cv.VideoWriter_fourcc(*"H264")
	videoWriter = cv.VideoWriter(outputPath, fourcc, 25, (width, height))


	for step in replay.Steps:
		# numpy array to image
		image = cv.cvtColor(step.HumanState, cv.COLOR_RGB2BGR)
		for i in range(frameDups):
			videoWriter.write(image)

	videoWriter.release()
	return

def CreateImage(replay:ER.EpisodeReplay, step:int) -> None:

	outputPath = os.path.join("src", "WebServer", "static", "assets", f"{replay.EpisodeId}_{step}.png")

	# numpy array to image
	image = cv.cvtColor(replay.Steps[step].HumanState, cv.COLOR_RGB2BGR)
	cv.imwrite(outputPath, image)

	return

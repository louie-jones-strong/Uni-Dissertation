import os
import src.Common.EpisodeReplay.EpisodeReplay as ER
import cv2 as cv

def CleanUpAssets() -> None:
	assetsFolder = os.path.join("src", "WebServer", "static", "assets")
	if os.path.exists(assetsFolder):
		for file in os.listdir(assetsFolder):
			os.remove(os.path.join(assetsFolder, file))
	else:
		os.makedirs(assetsFolder)

	return

def CreateVideo(replay:ER.EpisodeReplay) -> None:

	outputPath = os.path.join("src", "WebServer", "static", "assets", f"{replay.EpisodeId}.mp4")

	if os.path.exists(outputPath):
		return

	firstFrame = replay.Steps[0].HumanState

	if firstFrame is None:
		return

	width = firstFrame.shape[1]
	height = firstFrame.shape[0]
	frameDups = 3


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

	if os.path.exists(outputPath):
		return

	# numpy array to image
	humanState = replay.Steps[step].HumanState
	if humanState is None:
		return

	image = cv.cvtColor(humanState, cv.COLOR_RGB2BGR)
	cv.imwrite(outputPath, image)

	return

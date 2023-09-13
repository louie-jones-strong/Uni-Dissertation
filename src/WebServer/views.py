from flask import Blueprint, render_template, redirect
import typing
import os
import src.Common.Utils.PathHelper as PathHelper
import src.Common.Utils.Config.ConfigHelper as ConfigHelper
import src.Common.EpisodeReplay.EpisodeReplay as ER
import src.WebServer.AssetCreator as AssetCreator
import json
import numpy as np
import math
import src.Common.Store.ExperienceStore.EsNumpy as EsNumpy
import logging

class NumpyEncoder(json.JSONEncoder):
	""" Custom encoder for numpy data types """
	def default(self, obj):

		if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
							np.int16, np.int32, np.int64, np.uint8,
							np.uint16, np.uint32, np.uint64)):

			return int(obj)

		elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
			return float(obj)

		elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
			return {'real': obj.real, 'imag': obj.imag}

		elif isinstance(obj, (np.ndarray,)):
			return obj.tolist()

		elif isinstance(obj, (np.bool_)):
			return bool(obj)

		elif isinstance(obj, (np.void)):
			return None

		return json.JSONEncoder.default(self, obj)

def Setup(envConfig) -> None:

	rootPath = PathHelper.GetRootPath()
	DataFolder = os.path.join(rootPath, "Data", envConfig["Name"])
	RunFolder = os.path.join(DataFolder, envConfig["Group"])
	ReplaysFolder = os.path.join(RunFolder, "replays")

	reverbClient = None
	try:
		import reverb
		reverbClient = reverb.Client(f"experience-store:{5001}")
	except ImportError:
		logging.warning("Reverb not installed")

	views = Blueprint("views", __name__)

	ActionLabels = {}
	for key, value in envConfig["AgentConfig"]["Human"]["Controls"].items():
		ActionLabels[value] = key

	# load replays to review

	ReplaysToReview = {}
	for behaviour in ["Human", "Curated"]:
		toReviewPath = os.path.join(RunFolder, f"ReplaysToReview_{behaviour}.json")

		if os.path.exists(toReviewPath):
			ReplaysToReview[behaviour] = ConfigHelper.LoadConfig(toReviewPath)





	def GetCommonData() -> typing.Dict[str, typing.Any]:
		data = {}

		if not os.path.exists(ReplaysFolder):
			os.makedirs(ReplaysFolder)

		replayFolders = {}
		agentFolders = os.listdir(ReplaysFolder)
		for i in range(len(agentFolders)):
			agentFolder = os.path.join(ReplaysFolder, agentFolders[i])
			agentReplays = os.listdir(agentFolder)

			replayFolders[agentFolders[i]] = agentReplays


		data["EpisodeReplays"] = replayFolders
		data["envConfig"] = envConfig

		data["ActionLabels"] = ActionLabels


		return data

	def GetMonteCarloData(actionData, data):
		actionReason = actionData.ActionReason

		if actionReason is None:
			return data

		if "Tree" not in actionReason:
			return data

		treeRoot = actionReason["Tree"]

		treeTable = [[
			"ID",
			"Parent",
			"size",
			"Avg Rewards (Colour)",
			"Counts",
			"TotalRewards",
			"State",
			"EpisodeStep",
			"Done",
			"ActionIdx",
			"Action"
		]]

		treeTable = GetMonteCarloTreeTable(treeTable, treeRoot, parentId=None)


		jsonTree = json.dumps(treeTable, indent=4,
				separators=(', ', ': '), ensure_ascii=False,
				cls=NumpyEncoder)

		data["monteCarloTreeTable"] = jsonTree
		return data

	def GetMonteCarloTreeTable(treeTable, treeNode, parentId=None):
		actionIdx = treeNode["ActionIdxTaken"]
		action = None
		if actionIdx is not None:
			action = ActionLabels[actionIdx]


		parentStr = ""
		id = f"{action}"
		if parentId is not None and parentId != "None":
			parentStr = str(parentId)
			id = f"{parentStr},{action}"

		size = math.log(treeNode["Counts"] + 1) * 10
		avgValue = treeNode["TotalRewards"] / (treeNode["Counts"] + 1)

		treeTable.append([
			id,
			parentId,
			size,
			avgValue,
			treeNode["Counts"],
			treeNode["TotalRewards"],
			treeNode["State"],
			treeNode["EpisodeStep"],
			treeNode["Done"],
			treeNode["ActionIdxTaken"],
			action])

		children = treeNode["Children"]
		if children is not None:
			for c in range(len(children)):
				child = children[c]
				treeTable = GetMonteCarloTreeTable(treeTable, child, parentId=id)

		return treeTable







# region endpoints
	@views.route("/")
	def Home() -> str:
		data = GetCommonData()
		return render_template("overview.html", **data)


	@views.route("/metrics")
	def Metrics() -> str:
		# redirect to wandb page
		return redirect("https://wandb.ai/louiej-s/Dissertation", code=302)

	@views.route("/config")
	def Config() -> str:
		data = GetCommonData()

		configFolder = os.path.join(rootPath, "Config")

		# config = ConfigHelper.LoadConfig(data["envConfig"]["ConfigPath"])
		data["envConfigStr"] = ConfigHelper.PrintConfig(data["envConfig"])

		mlConfigPath = os.path.join(configFolder, "MLConfig.json")
		config = ConfigHelper.LoadConfig(mlConfigPath)
		data["mlConfigStr"] = ConfigHelper.PrintConfig(config)

		return render_template("config.html", **data)


	@views.route("/episodereview")
	def EpisodesToReview() -> str:
		data = GetCommonData()

		return render_template("episodestoreview.html", **data)


	@views.route("/episodereview/<folder>/<episode>")
	def EpisodeReview(folder, episode) -> str:
		data = GetCommonData()

		if folder not in data["EpisodeReplays"]:
			return "folder Not found"

		if episode not in data["EpisodeReplays"][folder]:
			return "episode Not found"

		replayPath = os.path.join(ReplaysFolder, folder, episode)
		replay = ER.EpisodeReplay.LoadFromFolder(replayPath)

		data["replayData"] = replay
		data["episodeFolder"] = folder
		data["episodeId"] = episode


		# make video for replay
		AssetCreator.CreateVideo(replay)

		return render_template("episodereview.html", **data)


	@views.route("/episodereview/<folder>/<episode>/action/<action>")
	def ReviewAction(folder, episode, action) -> str:
		data = GetCommonData()

		if folder not in data["EpisodeReplays"]:
			return "folder Not found"

		if episode not in data["EpisodeReplays"][folder]:
			return "episode Not found"

		replayPath = os.path.join(ReplaysFolder, folder, episode)
		replay = ER.EpisodeReplay.LoadFromFolder(replayPath)

		action = int(action)
		action = min(action, len(replay.Steps) - 1)
		action = max(action, 0)

		data["replayData"] = replay
		data["episodeFolder"] = folder
		data["episodeId"] = episode
		data["actionData"] = replay.Steps[action]
		data["actionIndex"] = action

		data = GetMonteCarloData(replay.Steps[action], data)

		# make video for replay
		AssetCreator.CreateImage(replay, action)

		return render_template("actionreview.html", **data)



	@views.route("/feedback/<behaviour>/<predicted>")
	def Feedback(behaviour, predicted) -> str:
		data = GetCommonData()

		if behaviour not in ReplaysToReview:
			return "behaviour Not found"

		predicted = predicted.lower()
		if predicted == "new":
			for i in range(len(ReplaysToReview[behaviour])):
				ReplaysToReview[behaviour][i]["Predicted"] = None
			predicted = None

		elif predicted == "true":
			predicted = True
		elif predicted == "false":
			predicted = False
		else:
			predicted = None



		replayToReview = None
		index = 0
		# loop through all replays and get the ones that need reviewing
		for i in range(len(ReplaysToReview[behaviour])):
			replay = ReplaysToReview[behaviour][i]

			if replay["Predicted"] is None:

				if predicted is not None:
					ReplaysToReview[behaviour][i]["Predicted"] = predicted
					predicted = None
				else:
					replayToReview = replay
					index = i
					break

		if replayToReview is None:
			# save the replays to review
			toReviewPath = os.path.join(RunFolder, f"ReplaysToReview_{behaviour}.json")
			ConfigHelper.SaveConfig(ReplaysToReview[behaviour], toReviewPath)
			return "No replays to review"

		folder = replayToReview["AgentType"]

		replayInfos = []
		replayIdx = 0
		while f"Replay_{replayIdx}" in replayToReview:

			episode = replayToReview[f"Replay_{replayIdx}"]

			if episode is None:
				continue

			replayPath = os.path.join(ReplaysFolder, folder, episode)
			replay = ER.EpisodeReplay.LoadFromFolder(replayPath)

			# make video for replay
			AssetCreator.CreateVideo(replay)
			replayInfos.append({"Folder": folder, "EpisodeId": replay.EpisodeId})

			replayIdx += 1

		data["replayInfos"] = replayInfos
		data["behaviour"] = behaviour
		data["episodeFolder"] = folder
		data["episodeId"] = episode
		data["idx"] = index

		return render_template("feedback.html", **data)

	@views.route("/behaviourexample/<behaviour>/<folder>/<episode>")
	def MarkEpisodeAsBehaviourExample(behaviour, folder, episode) -> str:
		data = GetCommonData()

		if folder not in data["EpisodeReplays"]:
			return "folder Not found"

		if episode not in data["EpisodeReplays"][folder]:
			return "episode Not found"

		if behaviour not in ReplaysToReview:
			return "behaviour Not found"

		replay = ER.EpisodeReplay.LoadFromFolder(os.path.join(ReplaysFolder, folder, episode))

		# add replay to behaviour examples

		examplesSavePath = os.path.join(DataFolder, "examples", behaviour)

		experienceStore = EsNumpy.EsNumpy(examplesSavePath)

		experienceStore.Load()
		state = replay.Steps[0].AgentState
		# add replay to experience store
		for i in range(1, len(replay.Steps)):
			step = replay.Steps[i]

			nextState = step.AgentState
			action = step.Action
			reward = step.Reward
			terminated = step.Terminated
			truncated = step.Truncated
			actionValues = step.GetActionValues()
			experienceStore.AddTransition(state, action, reward, nextState, terminated, truncated, actionValues)
			state = nextState

		experienceStore.EmptyTransitionBuffer()
		return f"added to {behaviour} behaviour examples"


	@views.route("/renderAssets/<behaviour>")
	def RenderAssets(behaviour) -> str:
		data = GetCommonData()

		if behaviour not in ReplaysToReview:
			return "behaviour Not found"


		# loop through all replays and get the ones that need reviewing
		for i in range(len(ReplaysToReview[behaviour])):
			replayToReview = ReplaysToReview[behaviour][i]

			folder = replayToReview["AgentType"]

			replayIdx = 0
			while f"Replay_{replayIdx}" in replayToReview:

				episode = replayToReview[f"Replay_{replayIdx}"]

				if episode is None:
					continue

				replayPath = os.path.join(ReplaysFolder, folder, episode)
				replay = ER.EpisodeReplay.LoadFromFolder(replayPath)

				# make video for replay
				AssetCreator.CreateVideo(replay, folder=folder)

				replayIdx += 1

		return "Created assets"



# endregion endpoints

	@views.route("/api/saveTrajectories", methods=["POST"])
	def SaveTrajectories() -> str:
		if reverbClient is None:
			return "Reverb not installed"
		else:
			checkPointPath = reverbClient.checkpoint()
			logging.info(checkPointPath)
			return checkPointPath

		return

	return views
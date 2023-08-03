import time
import numpy as np
import os

class EpisodeReplayStep:
	def __init__(self, frame, humanState, agentState, reward, action, actionReason):

		self.Frame = frame
		self.HumanState = humanState
		self.AgentState = agentState

		self.Reward = reward

		self.Action = action
		self.ActionReason = actionReason

		self.CompletedTime = time.time_ns()
		return

	@classmethod
	def Load(cls, data, replayFolder):

		instance = cls(None, None, None, None, None, None)
		instance.Frame = data.get("Frame", None)
		instance.AgentState = data.get("AgentState", None)
		instance.Reward = data.get("Reward", None)
		instance.Action = data.get("Action", None)
		instance.ActionReason = data.get("ActionReason", None)
		instance.CompletedTime = data.get("CompletedTime", None)


		# load the human state
		framesFolder = os.path.join(replayFolder, "humanStates")

		# create the run path
		if not os.path.exists(framesFolder):
			os.makedirs(framesFolder)

		frameFilePath = os.path.join(framesFolder, f"{instance.Frame}.npy")
		instance.HumanState = np.load(frameFilePath)
		return instance

	def Save(self, replayFolder):
		dict = {}
		dict["Frame"] = self.Frame
		dict["AgentState"] = self.AgentState
		dict["Reward"] = self.Reward
		dict["Action"] = self.Action
		dict["ActionReason"] = self.ActionReason
		dict["CompletedTime"] = self.CompletedTime

		# save the human state
		framesFolder = os.path.join(replayFolder, "humanStates")

		# create the run path
		if not os.path.exists(framesFolder):
			os.makedirs(framesFolder)

		frameFilePath = os.path.join(framesFolder, f"{self.Frame}.npy")

		np.save(frameFilePath, self.HumanState)


		return dict

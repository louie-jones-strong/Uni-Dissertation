import time
import numpy as np
import os
import src.Common.Utils.PathHelper as PathHelper

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
		instance.Reward = data.get("Reward", None)
		instance.Action = data.get("Action", None)
		instance.ActionReason = data.get("ActionReason", None)
		instance.CompletedTime = data.get("CompletedTime", None)

		instance.AgentState = data.get("AgentState", None)
		if instance.AgentState is None:
			frameFilePath = os.path.join(replayFolder, "agentState", f"{instance.Frame}.npy")
			PathHelper.EnsurePathExists(frameFilePath)
			instance.AgentState = np.load(frameFilePath)



		# load the human state
		frameFilePath = os.path.join(replayFolder, "humanStates", f"{instance.Frame}.npy")
		PathHelper.EnsurePathExists(frameFilePath)
		instance.HumanState = np.load(frameFilePath, allow_pickle=True)
		return instance

	def Save(self, replayFolder):
		dict = {}
		dict["Frame"] = self.Frame
		dict["Reward"] = self.Reward
		dict["Action"] = self.Action
		dict["ActionReason"] = self.ActionReason
		dict["CompletedTime"] = self.CompletedTime

		if isinstance(self.AgentState, np.ndarray):
			frameFilePath = os.path.join(replayFolder, "agentState", f"{self.Frame}.npy")
			PathHelper.EnsurePathExists(frameFilePath)
			np.save(frameFilePath, self.AgentState)
		else:
			dict["AgentState"] = self.AgentState



		# save the human state
		frameFilePath = os.path.join(replayFolder, "humanStates", f"{self.Frame}.npy")
		PathHelper.EnsurePathExists(frameFilePath)
		np.save(frameFilePath, self.HumanState)
		return dict

	def StateValue(self) -> float:

		if self.ActionReason is None:
			return 0

		if "ActionValues" not in self.ActionReason:
			return 0

		actionValues = self.ActionReason["ActionValues"]

		minValue = min(actionValues)
		maxValue = max(actionValues)
		return minValue, maxValue

import time
import numpy as np

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
	def FromJson(cls, data):

		instance = cls(None, None, None, None, None, None)
		instance.Frame = data.get("Frame", None)
		instance.HumanState = data.get("HumanState", None)
		instance.HumanState = np.array(instance.HumanState)

		instance.AgentState = data.get("AgentState", None)

		instance.Reward = data.get("Reward", None)

		instance.Action = data.get("Action", None)
		instance.ActionReason = data.get("ActionReason", None)

		instance.CompletedTime = data.get("CompletedTime", None)

		return instance

	def __dict__(self):

		return {
			"Frame": self.Frame,
			"HumanState": self.HumanState.tolist(),
			"AgentState": self.AgentState,
			"Reward": self.Reward,
			"Action": self.Action,
			"ActionReason": self.ActionReason,
			"CompletedTime": self.CompletedTime
		}

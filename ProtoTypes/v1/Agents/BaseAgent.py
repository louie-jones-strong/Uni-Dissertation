import enum
import os
import json


class AgentMode(enum.Enum):
	Train = 0
	Play = 1

def GetAgent(agentName):
	from . import RandomAgent, DQNAgent
	lookUp = {
		"Random": RandomAgent.RandomAgent,
		"DQN": DQNAgent.DQNAgent,

	}

	if agentName not in lookUp:
		return None

	return lookUp[agentName]




class BaseAgent:
	def __init__(self, env, replayBuffer, mode=AgentMode.Train):
		self.Env = env
		self.ReplayBuffer = replayBuffer
		self.Mode = mode
		self.Name = self.__class__.__name__.replace("Agent", "")

		self.Config = self.LoadConfig()
		self.FrameNum = 0
		self.TotalFrameNum = 0
		self.EpisodeNum = 0
		return

	def LoadConfig(self):
		config = {}

		path = os.path.join(os.path.abspath(os.curdir), "Config", f"AgentConfig_{self.Name}.json")
		if os.path.exists(path):
			with open(path, "r") as f:
				config = json.load(f)

		print(f"Agent config: {config}, path: {path}")

		return config

	def Reset(self):
		self.FrameNum = 0
		self.EpisodeNum += 1
		return


	def GetAction(self, state):
		self.FrameNum += 1
		self.TotalFrameNum += 1

		actionValues = self.GetActionValues(state)
		if actionValues is not None:
			return actionValues.argmax()

		return None

	def GetActionValues(self, state):
		return None

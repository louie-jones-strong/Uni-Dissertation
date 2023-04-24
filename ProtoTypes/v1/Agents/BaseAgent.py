import Utils.ReplayBuffer as ReplayBuffer
import enum
import os
import json


class AgentMode(enum.Enum):
	Train = 0
	Play = 1

def GetAgent(agentName):
	from . import RandomAgent, DQNAgent, HumanAgent
	lookUp = {
		"Random": RandomAgent.RandomAgent,
		"DQN": DQNAgent.DQNAgent,
		"Human": HumanAgent.HumanAgent,
	}

	if agentName not in lookUp:
		return None

	return lookUp[agentName]




class BaseAgent:
	def __init__(self, env, envConfig, mode=AgentMode.Train):
		self.Env = env
		self.Mode = mode
		self.Name = self.__class__.__name__.replace("Agent", "")


		self.TransitionAcc = ReplayBuffer.TransitionAccumulator(1000)

		self.LoadConfig(envConfig)

		self.FrameNum = 0
		self.TotalFrameNum = 0
		self.EpisodeNum = 0
		return

	def LoadConfig(self, envConfig):
		self.Config = {}

		path = os.path.join(os.path.abspath(os.curdir), "Config", f"AgentConfig_{self.Name}.json")
		if os.path.exists(path):
			with open(path, "r") as f:
				self.Config = json.load(f)

		self.EnvConfig = envConfig.get("AgentConfig", {}).get(self.Name, None)
		return

	def Reset(self):
		self.FrameNum = 0
		self.EpisodeNum += 1
		self.TransitionAcc.Clear()
		return

	def Remember(self, state, action, reward, nextState, done):
		self.TransitionAcc.Add(state, action, reward, nextState, done)
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

	def Save(self):
		return

	def Load(self):
		return

import enum
import random

import numpy as np
import src.Common.DataManager.DataManager as DataManager
import src.Common.Utils.Metrics.Logger as Logger
import src.Common.Utils.SharedCoreTypes as SCT
from numpy.typing import NDArray
import src.Common.Agents.ForwardModel as ForwardModel
import src.Common.Utils.ConfigHelper as ConfigHelper
from src.Common.Enums.AgentType import AgentType
from src.Common.Enums.PlayMode import PlayMode
from gymnasium.spaces import Discrete, Box
from typing import Union


def GetAgent(agentType:AgentType,
		overrideConfig:SCT.Config,
		isTrainingMode:bool) -> object:

	if agentType == AgentType.Random:
		from . import RandomAgent
		return RandomAgent.RandomAgent(overrideConfig, isTrainingMode)

	elif agentType == AgentType.Human:
		from . import HumanAgent
		return HumanAgent.HumanAgent(overrideConfig, isTrainingMode)

	# elif agentType == "DQN":
	# 	from . import DQNAgent
	# 	return DQNAgent.DQNAgent(overrideConfig, isTrainingMode)


	# elif agentType == "MonteCarlo":
	# 	from . import MonteCarloAgent
	# 	return MonteCarloAgent.MonteCarloAgent(overrideConfig, isTrainingMode, forwardModel)

	raise Exception(f"Agent \"{agentType}\" not found")

def ConfigToSpace(config:SCT.Config) -> Union[Discrete, Box]:

	if config["Type"] == "Discrete":
		space = Discrete(config["Shape"])
	elif config["Type"] == "Box":
		space = Box(config["Low"], config["High"], config["Shape"], config["Dtype"])

	return space



class BaseAgent(ConfigHelper.ConfigurableClass):
	def __init__(self, envConfig:SCT.Config, isTrainingMode:bool):
		self.LoadConfig(envConfig)
		self.Mode = PlayMode.Train if isTrainingMode else PlayMode.Play

		self.ObservationSpace = ConfigToSpace(envConfig["ObservationSpace"])
		self.ActionSpace = ConfigToSpace(envConfig["ActionSpace"])
		self.StepRewardRange = envConfig["StepRewardRange"]
		self.EpisodeRewardRange = envConfig["EpisodeRewardRange"]
		self.IsDeterministic = envConfig["IsDeterministic"]


		self._Logger = Logger.Logger()

		self.StepNum = 0
		self.TotalStepNum = 0
		self.TotalRememberedStep = 0
		self.EpisodeNum = 0
		return



	def Reset(self) -> None:
		self.StepNum = 0
		self.EpisodeNum += 1


		if self.Mode == PlayMode.Eval:
			self.Mode = PlayMode.Train
		elif self.Mode == PlayMode.Train:

			episodesBetweenEval = self.Config.get("EpisodesBetweenEval", -1)
			if episodesBetweenEval > 0 and self.EpisodeNum % episodesBetweenEval == 0:
				self.Mode = PlayMode.Eval
		return

	def Remember(self,
			state:SCT.State,
			action:SCT.Action,
			reward:SCT.Reward,
			nextState:SCT.State,
			terminated:bool,
			truncated:bool) -> None:

		self.TotalRememberedStep += 1
		return


	def GetAction(self, state:SCT.State) -> SCT.Action:
		self.StepNum += 1
		self.TotalStepNum += 1

		return 0

	def GetActionValues(self, state:SCT.State) -> NDArray[np.float32]:
		shape = SCT.JoinTuples(self.ActionSpace.shape, None)
		return np.ones(shape, dtype=np.float32)

	def _GetMaxValues(self, values:NDArray[np.float32]) -> int:
		maxValue = np.max(values)
		maxValues = np.where(values == maxValue)[0]
		choice = random.choice(maxValues)
		return int(choice)


	def Save(self, path:str) -> None:
		return

	def Load(self, path:str) -> None:
		return

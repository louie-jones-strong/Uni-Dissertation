import random

import numpy as np
import src.Common.Utils.Metrics.Logger as Logger
import src.Common.Utils.SharedCoreTypes as SCT
from numpy.typing import NDArray
import src.Common.Utils.ConfigHelper as ConfigHelper
from src.Common.Enums.eAgentType import eAgentType
from src.Common.Enums.ePlayMode import ePlayMode
from gymnasium.spaces import Discrete, Box
import typing
import src.Common.Agents.Models.ForwardModel as ForwardModel



def GetAgent(eAgentType:eAgentType,
		overrideConfig:SCT.Config,
		isTrainingMode:bool) -> object:

	if eAgentType == eAgentType.Random:
		from . import RandomAgent
		return RandomAgent.RandomAgent(overrideConfig, isTrainingMode)

	elif eAgentType == eAgentType.Human:
		from . import HumanAgent
		return HumanAgent.HumanAgent(overrideConfig, isTrainingMode)

	elif eAgentType == eAgentType.ML:
		from . import MonteCarloAgent

		forwardModel = ForwardModel.ForwardModel()
		return MonteCarloAgent.MonteCarloAgent(overrideConfig, isTrainingMode, forwardModel)

	# elif eAgentType == eAgentType.HardCoded:
	# 	from . import HardCodedAgent
	# 	return HardCodedAgent.HardCodedAgent(overrideConfig, isTrainingMode)



	raise Exception(f"Agent \"{eAgentType}\" not found")
	return

class BaseAgent(ConfigHelper.ConfigurableClass):
	def __init__(self, envConfig:SCT.Config, isTrainingMode:bool):
		self.LoadConfig(envConfig)
		self.EnvConfig = envConfig
		self.Mode = ePlayMode.Train if isTrainingMode else ePlayMode.Play

		self.ObservationSpace = ConfigHelper.ConfigToSpace(self.EnvConfig["ObservationSpace"])
		self.ActionSpace = ConfigHelper.ConfigToSpace(self.EnvConfig["ActionSpace"])
		self.StepRewardRange = self.EnvConfig["StepRewardRange"]
		self.EpisodeRewardRange = self.EnvConfig["EpisodeRewardRange"]
		self.IsDeterministic = self.EnvConfig["IsDeterministic"]

		self.ActionList = self._GetActionList()

		self.UpdateModels()


		self._Logger = Logger.Logger()

		self.StepNum = 0
		self.TotalStepNum = 0
		self.TotalRememberedStep = 0
		self.EpisodeNum = 0
		return

	def UpdateModels(self) -> None:

		return



	def Reset(self) -> None:
		self.StepNum = 0
		self.EpisodeNum += 1


		if self.Mode == ePlayMode.Eval:
			self.Mode = ePlayMode.Train
		elif self.Mode == ePlayMode.Train:

			episodesBetweenEval = self.Config.get("EpisodesBetweenEval", -1)
			if episodesBetweenEval > 0 and self.EpisodeNum % episodesBetweenEval == 0:
				self.Mode = ePlayMode.Eval
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

	def _GetActionList(self) -> typing.List[SCT.Action]:
		if isinstance(self.ActionSpace, Discrete):
			return [i for i in range(self.ActionSpace.n)]

		elif isinstance(self.ActionSpace, Box):
			raise NotImplementedError
		else:
			raise NotImplementedError

		return

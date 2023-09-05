import random

import numpy as np
import src.Common.Utils.Metrics.Logger as Logger
import src.Common.Utils.SharedCoreTypes as SCT
from numpy.typing import NDArray
import src.Common.Utils.Config.ConfigHelper as ConfigHelper
from src.Common.Enums.eAgentType import eAgentType
from src.Common.Enums.ePlayMode import ePlayMode
from gymnasium.spaces import Discrete, Box
import typing
import src.Worker.Agents.Models.ForwardModel as ForwardModel
import src.Worker.Agents.Models.ValueModel as ValueModel
import src.Worker.Agents.Models.PlayStyleModel as PlayStyleModel
import src.Worker.Environments.BaseEnv as BaseEnv
from src.Common.Enums.eModelType import eModelType

import os
from src.Common.Utils.PathHelper import GetRootPath
from src.Common.Utils.Config.ConfigurableClass import ConfigurableClass


def GetAgent(eAgentType:eAgentType,
		overrideConfig:SCT.Config,
		isTrainingMode:bool) -> 'BaseAgent':

	if eAgentType == eAgentType.Random:
		from . import RandomAgent
		return RandomAgent.RandomAgent(overrideConfig, isTrainingMode)

	elif eAgentType == eAgentType.Human:
		from . import HumanAgent
		return HumanAgent.HumanAgent(overrideConfig, isTrainingMode)

	elif eAgentType == eAgentType.ML:
		from . import MonteCarloAgent

		forwardModel = ForwardModel.ForwardModel()
		valueModel = ValueModel.ValueModel()

		humanLikeModel = PlayStyleModel.PlayStyleModel(eModelType.Human_Discriminator)
		playStyleModel = PlayStyleModel.PlayStyleModel(eModelType.PlayStyle_Discriminator)

		playStyleModels = {
			"Human": humanLikeModel,
			"Curated": playStyleModel
		}

		return MonteCarloAgent.MonteCarloAgent(overrideConfig, isTrainingMode, forwardModel, valueModel, playStyleModels)

	elif eAgentType == eAgentType.HardCoded:
		import importlib.util

		def ImportPath(name:str, path:str):
			spec = importlib.util.spec_from_file_location(name, path)
			module = importlib.util.module_from_spec(spec)
			spec.loader.exec_module(module)
			return module

		fileName = overrideConfig["Name"] + "Ai.py"
		agentPath = os.path.join(GetRootPath(), "src", "Worker", "Agents", "HardCoded", fileName)

		agent = ImportPath("HardCodedAgent", agentPath)

		return agent.HardCodedAi(overrideConfig, isTrainingMode)



	raise Exception(f"Agent \"{eAgentType}\" not found")
	return

class BaseAgent(ConfigurableClass):
	def __init__(self, envConfig:SCT.Config, isTrainingMode:bool):
		self.LoadConfig()
		self.EnvConfig = envConfig
		self.Mode = ePlayMode.Train if isTrainingMode else ePlayMode.Play

		self.ObservationSpace = ConfigHelper.ConfigToSpace(self.EnvConfig["ObservationSpace"])
		self.ActionSpace = ConfigHelper.ConfigToSpace(self.EnvConfig["ActionSpace"])
		self.StepRewardRange = self.EnvConfig["StepRewardRange"]
		self.EpisodeRewardRange = self.EnvConfig["EpisodeRewardRange"]
		self.IsDeterministic = self.EnvConfig["IsDeterministic"]

		self.ActionList = self._GetActionList()


		self._Logger = Logger.Logger()

		self.StepNum = 0
		self.TotalStepNum = 0
		self.EpisodeNum = 0

		self.ActionHistory = []
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

		self.ActionHistory.clear()
		return

	def Remember(self,
			state:SCT.State,
			action:SCT.Action,
			reward:SCT.Reward,
			nextState:SCT.State,
			terminated:bool,
			truncated:bool) -> None:
		self.ActionHistory.append(action)

		return


	def GetAction(self, state:SCT.State, env:BaseEnv.BaseEnv) -> typing.Tuple[SCT.Action, str]:
		self.StepNum += 1
		self.TotalStepNum += 1

		return 0, "BaseAgent"

	def GetActionValues(self, state:SCT.State, env:BaseEnv.BaseEnv) -> typing.Tuple[NDArray[np.float32], str]:
		shape = SCT.JoinTuples(self.ActionSpace.shape, None)
		return np.ones(shape, dtype=np.float32), "BaseAgent"

	@staticmethod
	def _GetMaxValues(values:NDArray[np.float32]) -> int:
		maxValue = np.max(values)
		maxValues = np.where(values == maxValue)[0]
		choice = random.choice(maxValues)
		return int(choice)

	@staticmethod
	def _SoftMaxSelection(values:NDArray[np.float32], temperature:float) -> int:
		probabilities = BaseAgent._SoftMax(values, temperature)
		choice = np.random.choice(len(values), p=probabilities)
		return int(choice), probabilities

	@staticmethod
	def _SoftMax(values:NDArray[np.float32], temperature:float) -> NDArray[np.float32]:

		x = values.copy()
		x[x <= 0] = 0

		if x.max() == 0:
			return np.ones(len(x)) / len(x)


		# normalize so that the max value is 1
		x = x / x.max()

		# apply temperature
		x = x / temperature

		e_x = np.exp(x)
		softMax = e_x / e_x.sum(axis=0)

		# set all values <= 0 to 0
		if x.min() <= 0 and x.max() > 0:
			softMax[x <= 0] = 0
			softMax = softMax / softMax.sum(axis=0)

		if softMax.max() == 0:
			return np.ones(len(x)) / len(x)

		return softMax



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

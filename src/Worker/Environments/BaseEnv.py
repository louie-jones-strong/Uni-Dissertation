import src.Common.Utils.SharedCoreTypes as SCT

import gymnasium as gym
import numpy as np
from copy import deepcopy
import typing

def GetEnv(envConfig:SCT.Config) -> 'BaseEnv':
	envType = envConfig["EnvType"]

	if envType == "Gym":

		import src.Worker.Environments.GymEnv as GymEnv
		return GymEnv.GymEnv(envConfig)

	raise Exception(f"EnvType \"{envType}\" not found")
	return




class BaseEnv:
	def __init__(self, envConfig:SCT.Config):
		self._Config = envConfig

		self.ObservationSpace:SCT.StateSpace = gym.spaces.Box(low=0, high=1, shape=(1, 2), dtype=np.uint8)
		self.ActionSpace:SCT.ActionSpace = gym.spaces.Discrete(1)
		self.RewardRange:typing.Tuple[float, float] = (0,0)

		self._CurrentStep = 0
		self._Done = False
		return

	def Clone(self) -> 'BaseEnv':
		return deepcopy(self)




	def Step(self, action:SCT.Action) -> typing.Tuple[SCT.State, SCT.Reward, bool, bool, SCT.Config]:
		"""
		:param action:
		:return: nextState, reward, done
		"""
		self._CurrentStep += 1


		state = 1
		reward = 0.0
		terminated = False
		truncated = False
		self._Done = terminated or truncated

		info = {}

		return state, reward, terminated, truncated, info

	def Reset(self) -> SCT.State:
		self._CurrentStep = 0
		self._Done = False

		state = 1
		return state



	def Render(self) -> None:
		return
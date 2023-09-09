from collections import deque
import src.Common.Utils.SharedCoreTypes as SCT
import typing
import numpy as np
from src.Common.Utils.Config.ConfigurableClass import ConfigurableClass

class EsBase(ConfigurableClass):
	def __init__(self) -> None:
		self.LoadConfig()

		self._TransitionBuffer = deque()
		self._TotalReward = 0
		return

	def AddTransition(self,
			state:SCT.State,
			nextState:SCT.State,
			action:SCT.Action, reward:SCT.Reward, terminated:bool, truncated:bool) -> None:

		transition = (state, nextState, action, reward, terminated, truncated)
		self._TransitionBuffer.append(transition)
		self._TotalReward += reward

		return

	def PopTransition(self) -> typing.Tuple[SCT.State, SCT.State, SCT.Action, SCT.Reward, SCT.Reward, bool, bool]:

		if len(self._TransitionBuffer) == 0:
			return None

		transition = self._TransitionBuffer.pop()
		state, nextState, action, reward, terminated, truncated = transition

		if isinstance(state, np.ndarray):
			state = state.astype(np.double)
			nextState = nextState.astype(np.double)


		self._TotalReward -= reward  # todo discount factor

		if len(self._TransitionBuffer) == 0:
			assert abs(self._TotalReward) <= 0.000_0001, f"TotalReward:{self._TotalReward}"
			self._TotalReward = 0.0

		return state, nextState, action, reward, self._TotalReward, terminated, truncated

	def EmptyTransitionBuffer(self) -> None:
		self._TransitionBuffer.clear()
		self._TotalReward = 0
		return
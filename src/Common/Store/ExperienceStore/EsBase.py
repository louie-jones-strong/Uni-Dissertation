from collections import deque
import src.Common.Utils.SharedCoreTypes as SCT

class EsBase():
	def __init__(self) -> None:

		self._TransitionBuffer = deque()
		self._TotalReward = 0
		self._DiscountFactor = 0.99  # todo make this configurable
		return

	def AddTransition(self,
			state:SCT.State,
			nextState:SCT.State,
			action:SCT.Action, reward:SCT.Reward, terminated:bool, truncated:bool) -> None:

		transition = (state, nextState, action, reward, terminated, truncated)
		self._TransitionBuffer.append(transition)
		self._TotalReward += reward

		return

	def EmptyTransitionBuffer(self) -> None:
		self._TransitionBuffer.clear()
		self._TotalReward = 0
		return
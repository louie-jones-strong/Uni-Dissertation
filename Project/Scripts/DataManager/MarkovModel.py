from typing import Optional
import Utils.SharedCoreTypes as SCT
from numpy.typing import NDArray

from . import State
import numpy as np


class MarkovModel:

	def __init__(self, numActions:int):
		self.NumActions = numActions
		self.States:dict[int, State.State] = {}
		return


	def GetStateInfo(self, state:SCT.State) -> tuple[NDArray[np.float32], NDArray[np.float32]]:

		stateItem = self._GetState(state)

		novelties = stateItem.GetActionNovelties()
		values = stateItem.GetActionValues()

		return novelties, values

	def Predict(self,
			state:SCT.State,
			action:SCT.Action
			) -> Optional[tuple[SCT.State, float, bool, bool]]:

		stateInfo = self._GetState(state)

		if stateInfo is None:
			return None

		if stateInfo.ActionCounts[action] == 0:
			return None

		nextState = self.States[stateInfo.NextStates[action]].RawState
		reward = stateInfo.ActionRewards[action]
		terminated = stateInfo.ActionTerminateds[action]
		truncated = False
		return nextState, reward, terminated, truncated


	def Remember(self,
			state:SCT.State,
			action:SCT.Action,
			reward:SCT.Reward,
			nextState:SCT.State,
			terminated:bool,
			truncated:bool) -> None:

		stateItem = self._GetState(state)
		stateItem.Remember(action, self._GetStateId(nextState), terminated, reward, self)
		return

	def OnEmptyTransAcc(self,
			state:SCT.State,
			action:SCT.Action,
			reward:SCT.Reward,
			nextState:SCT.State,
			terminated:bool,
			truncated:bool,
			qValue:float) -> None:

		stateItem = self._GetState(state)
		stateItem.Update(action, qValue, self)
		return


	def _GetState(self, state:SCT.State) -> State.State:
		stateId = self._GetStateId(state)

		if stateId not in self.States:
			stateItem = State.State(stateId, state, self.NumActions)
			self.States[stateId] = stateItem
			return stateItem

		return self.States[stateId]

	def _GetStateId(self, state:SCT.State) -> int:

		if isinstance(state, int):
			return state
		elif isinstance(state, np.ndarray):
			return hash(tuple(state.flatten()))

		return state.__hash__()


	def Save(self, path:str) -> None:

		return

	def Load(self, path:str) -> None:

		return
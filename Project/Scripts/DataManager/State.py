#region typing dependencies
from typing import TYPE_CHECKING, Any, Optional, Type, TypeVar

import Utils.SharedCoreTypes as SCT

from numpy.typing import NDArray
if TYPE_CHECKING:
	from DataManager.MarkovModel import MarkovModel
	pass
# endregion

# other file dependencies
import numpy as np

class State:
	def __init__(self, stateId:int, rawState:SCT.State, actionNum:int):
		self.StateId = stateId
		self.RawState = rawState

		self.FullyExplored = np.zeros(actionNum)

		self.ActionCounts = np.zeros(actionNum)
		self.NextStates = np.empty(actionNum)
		self.ActionTerminateds = np.zeros(actionNum)
		self.ActionRewards = np.zeros(actionNum)
		self.ActionTotalRewards = np.zeros(actionNum)
		return

	def Remember(self,
			action:SCT.Action,
			nextStateId:int,
			terminated:bool,
			reward:SCT.Reward,
			markovModel:"MarkovModel") -> None:

		self.ActionCounts[action] += 1
		self.NextStates[action] = nextStateId
		self.ActionTerminateds[action] = int(terminated == True)
		self.ActionRewards[action] = reward

		self._CheckExplored(action, markovModel)
		return

	def Update(self, action:SCT.Action, totalReward:SCT.Reward, markovModel:"MarkovModel") -> None:
		self.ActionTotalRewards[action] += totalReward
		self._CheckExplored(action, markovModel)
		return

	def _CheckExplored(self, action:SCT.Action, markovModel:"MarkovModel") -> None:

		if self.FullyExplored[action] == 1:
			return

		if self.ActionTerminateds[action] >= 1:
			self.FullyExplored[action] = 1
			return


		nextStateId = self.NextStates[action]
		if nextStateId not in markovModel.States:
			return


		nextState = markovModel.States[nextStateId]

		# check all actions are fully explored
		fullyExplored = (nextState.ActionCounts > 0).all()
		self.FullyExplored[action] = fullyExplored
		return



	def GetActionValues(self) -> NDArray[np.float32]:
		counts = self.ActionCounts

		if sum(counts) == 0:
			return self.GetActionNovelties()

		counts = np.where(counts == 0, 1, counts)

		avgRewards = self.ActionTotalRewards / counts
		avgRewards += self.ActionRewards
		return avgRewards

	def GetActionNovelties(self) -> NDArray[np.float32]:

		countMax = self.ActionCounts.max()
		if countMax == 0:
			return np.ones_like(self.ActionCounts)

		novelty:NDArray[np.float32] = 1.1-(self.ActionCounts / countMax)

		# set all actions that transition to the same state to non novel
		novelty *= 1 - (self.NextStates == self.StateId)


		# set all actions that end the episode to non novel
		endActions = self.ActionTerminateds * (self.ActionRewards <= 0.0)
		novelty *= (1 - endActions)

		# all actions that have been fully explored are not novel
		novelty *= (1 - self.FullyExplored)

		return novelty
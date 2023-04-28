from . import BaseAgent
import numpy as np

class ExplorationAgent(BaseAgent.BaseAgent):

	def __init__(self, env, config):
		super().__init__(env, config)

		self.MdpGraph = {}
		return


	def GetActionValues(self, state):
		novelties = self._GetState(state).GetActionNovelties()
		return novelties


	def Remember(self, state, action, reward, nextState, terminated, truncated):
		super().Remember(state, action, reward, nextState, terminated, truncated)

		stateItem = self._GetState(state)
		stateItem.Remember(action, self._GetStateId(nextState), terminated, reward)
		return


	def _GetState(self, state):
		stateId = self._GetStateId(state)

		if stateId not in self.MdpGraph:
			stateItem = State(stateId, self.Env.ActionSpace.n)
			self.MdpGraph[stateId] = stateItem
			return stateItem

		return self.MdpGraph[stateId]


	def _GetStateId(self, state):

		if isinstance(state, int):
			return state

		return hash(tuple(state.flat))


class State:
	def __init__(self, stateId, actionNum):
		self.StateId = stateId

		self.ActionCounts      = np.zeros(actionNum)
		self.NextStates        = np.empty(actionNum)
		self.ActionTerminateds = np.zeros(actionNum)
		self.ActionRewards     = np.zeros(actionNum)
		return

	def Remember(self, action, nextStateId, terminated, reward):
		self.ActionCounts[action]     += 1
		self.NextStates[action]        = nextStateId
		self.ActionTerminateds[action] = int(terminated == True)
		self.ActionRewards[action]     = reward
		return

	def GetActionNovelties(self):

		countSum = self.ActionCounts.max()
		if countSum == 0:
			return np.ones_like(self.ActionCounts)

		novelty = 1 - (self.ActionCounts / countSum)

		# set all actions that transition to the same state to non novel
		novelty *= 1 - (self.NextStates == self.StateId)


		# set all actions that end the episode to non novel
		endActions = self.ActionTerminateds * (self.ActionRewards <= 0.0)
		novelty *= 1 - endActions
		return novelty


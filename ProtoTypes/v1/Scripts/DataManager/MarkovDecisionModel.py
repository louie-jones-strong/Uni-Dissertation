import numpy as np

class MarkovDecisionModel:

	def __init__(self, numActions):
		self.NumActions = numActions
		self.States = {}
		self.StateLookup = {}
		return


	def GetStateInfo(self, state):

		state = self._GetState(state)

		novelties = state.GetActionNovelties()
		values = state.GetActionValues()

		return novelties, values

	def Predict(self, state, action):
		stateInfo = self._GetState(state)

		if stateInfo is None:
			return None, None, None, None

		if stateInfo.ActionCounts[action] == 0:
			return None, None, None, None

		nextState = self.StateLookup[stateInfo.NextStates[action]]
		reward = stateInfo.ActionRewards[action]
		terminated = stateInfo.ActionTerminateds[action]
		truncated = False
		return nextState, reward, terminated, truncated


	def Remember(self, state, action, reward, nextState, terminated, truncated):

		stateItem = self._GetState(state)
		stateItem.Remember(action, self._GetStateId(nextState), terminated, reward)
		return

	def OnEmptyTransAcc(self, state, action, reward, nextState, terminated, truncated, qValue):
		stateItem = self._GetState(state)
		stateItem.Update(action, qValue)
		return


	def _GetState(self, state):
		stateId = self._GetStateId(state)

		if stateId not in self.States:
			stateItem = State(stateId, self.NumActions)
			self.States[stateId] = stateItem
			self.StateLookup[stateId] = state
			return stateItem

		return self.States[stateId]

	def _GetStateId(self, state):

		if isinstance(state, int):
			return state

		return hash(tuple(state.flat))


	def Save(self, path):

		return

	def Load(self, path):

		return


class State:
	def __init__(self, stateId, actionNum):
		self.StateId = stateId

		self.ActionCounts       = np.zeros(actionNum)
		self.NextStates         = np.empty(actionNum)
		self.ActionTerminateds  = np.zeros(actionNum)
		self.ActionRewards      = np.zeros(actionNum)
		self.ActionTotalRewards = np.zeros(actionNum)
		return

	def Remember(self, action, nextStateId, terminated, reward):
		self.ActionCounts[action]       += 1
		self.NextStates[action]          = nextStateId
		self.ActionTerminateds[action]   = int(terminated == True)
		self.ActionRewards[action]       = reward
		self.ActionTotalRewards[action]  = reward
		return

	def Update(self, action, totalReward):
		self.ActionTotalRewards[action] += totalReward
		return



	def GetActionValues(self):
		counts = self.ActionCounts

		if sum(counts) == 0:
			return self.GetActionNovelties()

		counts = np.where(counts == 0, 1, counts)

		avgRewards = self.ActionTotalRewards / counts
		return avgRewards

	def GetActionNovelties(self):

		countMax = self.ActionCounts.max()
		if countMax == 0:
			return np.ones_like(self.ActionCounts)

		novelty = 1.1-(self.ActionCounts / countMax)

		# set all actions that transition to the same state to non novel
		novelty *= 1 - (self.NextStates == self.StateId)


		# set all actions that end the episode to non novel
		endActions = self.ActionTerminateds * (self.ActionRewards <= 0.0)
		novelty *= (1 - endActions)
		return novelty
import numpy as np

class MarkovModel:

	def __init__(self, numActions):
		self.NumActions = numActions
		self.States = {}
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

		nextState = self.States[stateInfo.NextStates[action]].RawState
		reward = stateInfo.ActionRewards[action]
		terminated = stateInfo.ActionTerminateds[action]
		truncated = False
		return nextState, reward, terminated, truncated


	def Remember(self, state, action, reward, nextState, terminated, truncated):

		stateItem = self._GetState(state)
		stateItem.Remember(action, self._GetStateId(nextState), terminated, reward, self)
		return

	def OnEmptyTransAcc(self, state, action, reward, nextState, terminated, truncated, qValue):
		stateItem = self._GetState(state)
		stateItem.Update(action, qValue, self)
		return


	def _GetState(self, state):
		stateId = self._GetStateId(state)

		if stateId not in self.States:
			stateItem = State(stateId, state, self.NumActions)
			self.States[stateId] = stateItem
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
	def __init__(self, stateId, rawState, actionNum):
		self.StateId = stateId
		self.RawState = rawState

		self.FullyExplored      = np.zeros(actionNum)

		self.ActionCounts       = np.zeros(actionNum)
		self.NextStates         = np.empty(actionNum)
		self.ActionTerminateds  = np.zeros(actionNum)
		self.ActionRewards      = np.zeros(actionNum)
		self.ActionTotalRewards = np.zeros(actionNum)
		return

	def Remember(self, action, nextStateId, terminated, reward, markovModel):
		self.ActionCounts[action]       += 1
		self.NextStates[action]          = nextStateId
		self.ActionTerminateds[action]   = int(terminated == True)
		self.ActionRewards[action]       = reward

		self._CheckExplored(action, markovModel)
		return

	def Update(self, action, totalReward, markovModel):
		self.ActionTotalRewards[action] += totalReward
		self._CheckExplored(action, markovModel)
		return

	def _CheckExplored(self, action, markovModel):

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



	def GetActionValues(self):
		counts = self.ActionCounts

		if sum(counts) == 0:
			return self.GetActionNovelties()

		counts = np.where(counts == 0, 1, counts)

		avgRewards = self.ActionTotalRewards / counts
		avgRewards += self.ActionRewards
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

		# all actions that have been fully explored are not novel
		novelty *= (1 - self.FullyExplored)

		return novelty
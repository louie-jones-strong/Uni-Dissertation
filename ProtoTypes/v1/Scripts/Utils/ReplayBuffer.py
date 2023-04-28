import collections
import numpy as np
from os import path, makedirs

#inspired by
# https://github.com/deepmind/dqn_zoo/blob/master/dqn_zoo/replay.py

class ReplayBuffer:
	def __init__(self, capacity, env):

		stateShape = env.ObservationSpace.shape
		stateType = env.ObservationSpace.dtype
		actionShape = env.ActionSpace.shape
		actionType = env.ActionSpace.dtype


		self.Capacity = capacity
		self.Count = 0
		self.Current = 0

		print("Replay state Size", (capacity,) + stateShape)
		print("Replay action Size", (capacity,) + actionShape)

		self._States        = np.empty((capacity,) + stateShape,  dtype=stateType)
		self._Actions       = np.empty((capacity,) + actionShape, dtype=actionType)
		self._Rewards       = np.empty((capacity),               dtype=np.float32)
		self._NextStates    = np.empty((capacity,) + stateShape,  dtype=stateType)
		self._Dones         = np.empty((capacity),               dtype=np.bool)
		self._FutureRewards = np.empty((capacity),               dtype=np.float32)
		self._Priorities    = np.zeros((capacity),               dtype=np.float32)

		return

	def Add(self, state, action, reward, nextState, done, futureReward=None):

		self._States[self.Current]        = state
		self._Actions[self.Current]       = action
		self._Rewards[self.Current]       = reward
		self._NextStates[self.Current]    = nextState
		self._Dones[self.Current]         = done
		self._FutureRewards[self.Current] = futureReward
		self._Priorities[self.Current]    = max(self._Priorities.max(), 1.0)


		self.Current += 1
		self.Count = max(self.Count, self.Current)
		self.Current = self.Current % self.Capacity
		return

	def Sample(self, batchSize, priorityScale=1.0):
		batchSize = min(batchSize, self.Count)

		scaledPriorities = self._Priorities[:self.Count] ** priorityScale
		probabilities = scaledPriorities / sum(scaledPriorities)

		indexs = np.random.choice(self.Count, batchSize, p=probabilities)

		states        = self._States[indexs]
		actions       = self._Actions[indexs]
		rewards       = self._Rewards[indexs]
		nextStates    = self._NextStates[indexs]
		dones         = self._Dones[indexs]
		futureRewards = self._FutureRewards[indexs]
		priorities    = self._Priorities[indexs]

		if None in futureRewards:
			futureRewards = None

		return indexs, states, actions, rewards, nextStates, dones, futureRewards, priorities

	def UpdatePriorities(self, indexs, priorities, offset=0.1):

		for i in range(len(indexs)):
			self._Priorities[indexs[i]] = priorities[i] + offset
		return


	def __len__(self):
		return self.Count

	def Save(self, folderPath):
		if not path.exists(folderPath):
			makedirs(folderPath)

		np.save(path.join(folderPath, "States.npy"), self._States)
		np.save(path.join(folderPath, "Actions.npy"), self._Actions)
		np.save(path.join(folderPath, "Rewards.npy"), self._Rewards)
		np.save(path.join(folderPath, "NextStates.npy"), self._NextStates)
		np.save(path.join(folderPath, "Dones.npy"), self._Dones)
		np.save(path.join(folderPath, "FutureRewards.npy"), self._FutureRewards)
		np.save(path.join(folderPath, "Priorities.npy"), self._Priorities)
		return

	def Load(self, folderPath):
		if not path.exists(folderPath):
			return

		self._States = np.load(path.join(folderPath, "States.npy"))
		self._Actions = np.load(path.join(folderPath, "Actions.npy"))
		self._Rewards = np.load(path.join(folderPath, "Rewards.npy"))
		self._NextStates = np.load(path.join(folderPath, "NextStates.npy"))
		self._Dones = np.load(path.join(folderPath, "Dones.npy"))
		self._FutureRewards = np.load(path.join(folderPath, "FutureRewards.npy"))
		self._Priorities = np.load(path.join(folderPath, "Priorities.npy"))
		return

class TransitionAccumulator:
	def __init__(self, capacity):
		self.Capacity = capacity
		self.Store = collections.deque(maxlen=capacity)
		return

	def Add(self, state, action, reward, newState, done):
		self.Store.append((state, action, reward, newState, done))

		if len(self.Store) > self.Capacity:
			self.Store.popleft(0)
		return

	def TransferToReplayBuffer(self, replayBuffer):
		totalReward = 0

		while len(self.Store) > 0:
			transition = self.Store.pop()

			# unpack transition
			state, action, reward, newState, done = transition

			replayBuffer.Add(state, action, reward, newState, done, futureReward=totalReward)
			totalReward += reward

		self.Clear()
		return

	def Clear(self):
		self.Store.clear()
		return

	def __len__(self):
		return len(self.Store)
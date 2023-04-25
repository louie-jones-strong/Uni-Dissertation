import collections
import numpy as np
import os
from os import path

#inspired by
# https://github.com/deepmind/dqn_zoo/blob/master/dqn_zoo/replay.py

class ReplayBuffer:
	def __init__(self, capacity, env):

		stateShape = env.observation_space.shape
		stateType = env.observation_space.dtype
		actionShape = env.action_space.shape
		actionType = env.action_space.dtype


		self.Capacity = capacity
		self.Count = 0
		self.Current = 0

		print("stateShape", stateShape)

		self._States		= np.empty((capacity,) + stateShape,  dtype=stateType)
		self._Actions		= np.empty((capacity,) + actionShape, dtype=actionType)
		self._Rewards		= np.empty((capacity),               dtype=np.float32)
		self._NextStates	= np.empty((capacity,) + stateShape,  dtype=stateType)
		self._Dones			= np.empty((capacity),               dtype=np.bool)
		self._FutureRewards	= np.empty((capacity),               dtype=np.float32)

		return

	def Add(self, state, action, reward, nextState, done, futureReward=None):

		self._States[self.Current]		= state
		self._Actions[self.Current]		= action
		self._Rewards[self.Current]		= reward
		self._NextStates[self.Current]	= nextState
		self._Dones[self.Current]		= done
		self._FutureRewards[self.Current]= futureReward


		self.Current += 1
		self.Count = max(self.Count, self.Current)
		self.Current = self.Current % self.Capacity
		return

	def Sample(self, batchSize):
		batchSize = min(batchSize, self.Count)

		indexs = np.random.choice(self.Count, batchSize)

		states		= self._States[indexs]
		actions		= self._Actions[indexs]
		rewards		= self._Rewards[indexs]
		nextStates	= self._NextStates[indexs]
		dones		= self._Dones[indexs]
		futureRewards= self._FutureRewards[indexs]

		if None in futureRewards:
			futureRewards = None

		return states, actions, rewards, nextStates, dones, futureRewards

	def __len__(self):
		return self.Count

	def Save(self, folderPath):
		if not path.exists(folderPath):
			os.makedirs(folderPath)

		np.save(path.join(folderPath, "States.npy"), self._States)
		np.save(path.join(folderPath, "Actions.npy"), self._Actions)
		np.save(path.join(folderPath, "Rewards.npy"), self._Rewards)
		np.save(path.join(folderPath, "NextStates.npy"), self._NextStates)
		np.save(path.join(folderPath, "Dones.npy"), self._Dones)
		np.save(path.join(folderPath, "FutureRewards.npy"), self._FutureRewards)
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
import collections
import numpy as np

#inspired by
# https://github.com/deepmind/dqn_zoo/blob/master/dqn_zoo/replay.py

class ReplayBuffer:
	def __init__(self, capacity):
		self.Capacity = capacity
		self.Store = []
		return

	def Add(self, state, action, reward, newState, done):
		self.Store.append((state, action, reward, newState, done))

		if len(self.Store) > self.Capacity:
			self.Store.pop(0)

		return

	def Sample(self, batchSize):
		batchSize = min(batchSize, len(self.Store))

		indexs = np.random.choice(len(self.Store), batchSize, replace=False)
		states, actions, rewards, newStates, dones = zip(*[self.Store[i] for i in indexs])
		return states, actions, rewards, newStates, dones

	def __len__(self):
		return len(self.Store)


class TransitionAccumulator:
	def __init__(self, capacity):
		self.Capacity = capacity
		self.Store = collections.deque(maxlen=capacity)
		return

	def Add(self, state, action, reward, newState, done):
		self.Store.append((state, action, reward, newState, done))
		return

	def TransferToReplayBuffer(self, replayBuffer):

		while len(self.Store) > 0:
			transition = self.Store.pop()
			replayBuffer.Add(*transition)

		self.Clear()
		return

	def Clear(self):
		self.Store.clear()
		return

	def __len__(self):
		return len(self.Store)
import Utils.Singleton as Singleton
from . import ReplayBuffer as ReplayBuffer
from . import MarkovModel as MarkovModel
from . import DataColumnTypes as DataColumnTypes
from events import Events
from collections import deque

class DataManager(Singleton.Singleton):
	def __init__(self):
		return

	def Setup(self, config, env):
		self.LoadConfig(config)
		self._Env = env

		# transition accumulator
		self._OnEmptyTransAcc = Events()
		self._TransitionAccumulator = deque()
		self._QValueAccumulator = 0

		self._ReplayBuffer = ReplayBuffer.ReplayBuffer(self._Config["ReplayBufferMaxSize"], self._Env)
		self._MarkovModel = MarkovModel.MarkovModel(self._Env.ActionSpace.n)

		return

	def LoadConfig(self, config):
		self._Config = config

		self._Config["ReplayBufferMaxSize"] = 100000
		self._Config["TransitionAccumulatorSize"] = 1000
		self._Config["QFuncGamma"] = 0.99

		return

	def Save(self, path):
		self._EmptyAccumulator()
		self._ReplayBuffer.Save(path)
		self._MarkovModel.Save(path)
		return

	def Load(self, path):
		self._ReplayBuffer.Load(path)
		self._MarkovModel.Load(path)
		return


	def Sample(self, columns, batchSize=-1, priorityKey=None, priorityScale=1.0):


		samples = self._ReplayBuffer.Sample(batchSize, priorityKey=priorityKey, priorityScale=priorityScale)
		indexs, states, actions, rewards, nextStates, terminateds, truncateds, futureRewards, priorities = samples

		rowsOrder = (states, nextStates, actions, rewards, futureRewards, terminateds, truncateds)
		columns = DataColumnTypes.GetColumn(columns, rowsOrder)

		return indexs, priorities, columns



	def SubToOnEmptyTransAcc(self, callback):
		self._OnEmptyTransAcc += callback
		return


	def EnvRemember(self, state, action, reward, nextState, terminated, truncated):

		# self._MarkovModel.Remember(state, action, reward, nextState, terminated, truncated)

		# add transition to the transition accumulator
		transition = (state, action, reward, nextState, terminated, truncated)
		self._QValueAccumulator += reward * (self._Config["QFuncGamma"] ** len(self._TransitionAccumulator))
		self._TransitionAccumulator.append(transition)


		# if transition accumulator is full, remove the oldest
		# transition and add it to the replay buffer
		if len(self._TransitionAccumulator) > self._Config["TransitionAccumulatorSize"]:
			self._PopAccumulator()

		assert len(self._TransitionAccumulator) <= self._Config["TransitionAccumulatorSize"], f"Transition accumulator has size of: {len(self._TransitionAccumulator)}"
		return

	def EnvReset(self):

		# empty transition accumulator into the replay buffer
		self._EmptyAccumulator()
		return


	def _EmptyAccumulator(self):
		while len(self._TransitionAccumulator) > 0:
			self._PopAccumulator()


		epsilon = 0.1 ** 10
		assert len(self._TransitionAccumulator) == 0, f"Transition accumulator not empty, has size of: {len(self._TransitionAccumulator)}"
		assert abs(self._QValueAccumulator) <= epsilon , f"Q value accumulator not empty, has value of: {self._QValueAccumulator}, {epsilon}"

		self._QValueAccumulator = 0
		return

	def _PopAccumulator(self):
		oldest = self._TransitionAccumulator.popleft()

		# unpack transition
		state, action, reward, nextState, terminated, truncated = oldest
		qValue = self._QValueAccumulator

		# invoke the event
		self._OnEmptyTransAcc.Invoke(state, action, reward, nextState, terminated, truncated, qValue)
		# self._MarkovModel.OnEmptyTransAcc(state, action, reward, nextState, terminated, truncated, qValue)

		# add to the replay buffer
		self._ReplayBuffer.Add(state, action, reward, nextState, terminated, truncated, futureReward=qValue)

		# update the q value accumulator
		self._QValueAccumulator -= reward
		self._QValueAccumulator *= (1/self._Config["QFuncGamma"])
		return


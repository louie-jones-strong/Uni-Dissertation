import os
import typing
from collections import deque
from typing import Optional

import numpy as np
import src.DataManager.DataColumnTypes as DCT
import src.DataManager.MarkovModel as MarkovModel
import src.DataManager.ReplayBuffer as ReplayBuffer
import src.Utils.SharedCoreTypes as SCT
import src.Utils.Singleton as Singleton
from numpy.typing import NDArray
from tensorflow.keras.utils import to_categorical
import gymnasium.spaces as spaces


class DataManager(Singleton.Singleton):

	def Setup(self,
			config:SCT.Config,
			observationSpace:SCT.StateSpace,
			actionSpace:SCT.ActionSpace) -> None:

		self.LoadConfig(config)
		self.ObservationSpace = observationSpace
		self.ActionSpace = actionSpace

		# transition accumulator
		stateTupleType = typing.Tuple[SCT.State, SCT.Action, SCT.Reward, SCT.State, bool, bool]
		self._TransitionAccumulator: typing.Deque[stateTupleType] = deque()
		self._QValueAccumulator:SCT.Reward = 0

		self._ReplayBuffer = ReplayBuffer.ReplayBuffer(
				self._Config["ReplayBufferMaxSize"],
				self.ObservationSpace,
				self.ActionSpace)

		self._MarkovModel = MarkovModel.MarkovModel(int(self.ActionSpace.n))

		return

	def LoadConfig(self, config:SCT.Config) -> None:
		self._Config = config

		self._Config["ReplayBufferMaxSize"] = 100000
		self._Config["TransitionAccumulatorSize"] = 1000
		self._Config["QFuncGamma"] = 0.99

		return

	def Save(self, path:str) -> None:
		self._EmptyAccumulator()

		self._ReplayBuffer.Save(os.path.join(path, "ReplayBuffer"))
		self._MarkovModel.Save(os.path.join(path, "MarkovModel"))

		return

	def Load(self, path:str) -> None:
		self._ReplayBuffer.Load(os.path.join(path, "ReplayBuffer"))
		self._MarkovModel.Load(os.path.join(path, "MarkovModel"))
		return


# region Calls from Env Runner

	def EnvRemember(self,
			state:SCT.State,
			action:SCT.Action,
			reward:SCT.Reward,
			nextState:SCT.State,
			terminated:bool,
			truncated:bool) -> None:

		# self._MarkovModel.Remember(state, action, reward, nextState, terminated, truncated)

		# add transition to the transition accumulator
		transition = (state, action, reward, nextState, terminated, truncated)
		self._QValueAccumulator += reward * (self._Config["QFuncGamma"] ** len(self._TransitionAccumulator))
		self._TransitionAccumulator.append(transition)


		# if transition accumulator is full, remove the oldest
		# transition and add it to the replay buffer
		if len(self._TransitionAccumulator) > self._Config["TransitionAccumulatorSize"]:
			self._PopAccumulator()

		assert len(self._TransitionAccumulator) <= self._Config["TransitionAccumulatorSize"], \
			f"Transition accumulator has size of: {len(self._TransitionAccumulator)}"

		return

	def EnvReset(self) -> None:

		# empty transition accumulator into the replay buffer
		self._EmptyAccumulator()
		return

# endregion

	def _EmptyAccumulator(self) -> None:
		while len(self._TransitionAccumulator) > 0:
			self._PopAccumulator()


		epsilon = 0.1 ** 9
		assert len(self._TransitionAccumulator) == 0, \
			f"Transition accumulator not empty, has size of: {len(self._TransitionAccumulator)}"

		assert abs(self._QValueAccumulator) <= epsilon, \
			f"Q value accumulator not empty, has value of: {self._QValueAccumulator}, {epsilon}"

		self._QValueAccumulator = 0
		return

	def _PopAccumulator(self) -> None:
		oldest = self._TransitionAccumulator.popleft()

		# unpack transition
		state, action, reward, nextState, terminated, truncated = oldest
		qValue = self._QValueAccumulator

		# self._MarkovModel.OnEmptyTransAcc(state, action, reward, nextState, terminated, truncated, qValue)

		# add to the replay buffer
		self._ReplayBuffer.Add(state, action, reward, nextState, terminated, truncated, futureReward=qValue)

		# update the q value accumulator
		self._QValueAccumulator -= reward
		self._QValueAccumulator *= (1/self._Config["QFuncGamma"])
		return





# region data sampling

	def GetColumns(self, columns:typing.List[DCT.DataColumnTypes]) -> typing.List[NDArray]:
		columnsData = self._ReplayBuffer.SampleAll()

		columnsData = DCT.FilterColumns(columns, columnsData)
		return columnsData

	def GetXYData(self,
			xColumns:typing.List[DCT.DataColumnTypes],
			yColumns:typing.List[DCT.DataColumnTypes]
			) -> typing.Tuple[typing.List[NDArray], typing.List[NDArray]]:

		joinedColumns = xColumns + yColumns
		joinedColumnsData = self.GetColumns(joinedColumns)

		xColumnsData = joinedColumnsData[:len(xColumns)]
		yColumnsData = joinedColumnsData[len(xColumns):]

		return xColumnsData, yColumnsData




	def GetSampleIndexs(self, batchSize:int,
			priorityKey:Optional[str] = None,
			priorityScale:float = 1.0) -> typing.Tuple[NDArray[np.int_], NDArray[np.float32]]:

		return self._ReplayBuffer.GetSampleIndexs(batchSize, priorityKey, priorityScale)

	def SampleArrays(self,
			arrays,
			batchSize:int,
			priorityKey:Optional[str] = None,
			priorityScale:float = 1.0):

		indexs, priorities = self.GetSampleIndexs(batchSize, priorityKey, priorityScale)

		sampledArrays = []
		for array in arrays:
			sampledArrays.append(array[indexs])

		return indexs, priorities, sampledArrays


# endregion



# region pre and post process columns
	def PreProcessColumns(self, columnsData, columnLabels):

		data = self.PreProcessSingleColumn(columnsData[0], columnLabels[0])

		for i in range(1, len(columnLabels)):
			columnData = self.PreProcessSingleColumn(columnsData[i], columnLabels[i])

			data = np.concatenate((data, columnData), axis=1)

		# data = np.array(data)
		# data = self._JoinColumnsData(data)

		return data

	def PostProcessColumns(self, columnsData, columnLabels):

		if len(columnLabels) == 1:
			return self.PostProcessSingleColumn(columnsData, columnLabels[0])

		data = []
		for i in range(len(columnLabels)):
			columnData = self.PostProcessSingleColumn(columnsData[i], columnLabels[i])
			data.append(columnData)

		data = np.array(data)
		data = self._JoinColumnsData(data)

		return data



	def _JoinColumnsData(self, columnsData):
		if len(columnsData) == 1:
			return columnsData[0]

		output = []
		for i in range(len(columnsData[0])):
			joinedRow = [columnsData[j][i] for j in range(len(columnsData))]
			output.append(np.array(joinedRow))

		output = np.array(output)
		return output




	def PreProcessSingleColumn(self, data, label):

		# add a dimension to the data at the end
		proccessed = np.reshape(data, (len(data), -1))

		if (label == DCT.DataColumnTypes.Terminated or
				label == DCT.DataColumnTypes.Truncated):
			# one hot encode the boolean values
			intBools = [int(i) for i in data]
			proccessed = to_categorical(intBools, num_classes=2)

		elif label == DCT.DataColumnTypes.Reward:
			# todo if reward is clipped then we can one hot encode it
			pass

		elif label == DCT.DataColumnTypes.Action and \
				isinstance(self.ActionSpace, spaces.Discrete):

			# one hot encode the action
			proccessed = to_categorical(data, num_classes=self.ActionSpace.n)

		elif (label == DCT.DataColumnTypes.CurrentState or
				label == DCT.DataColumnTypes.NextState) and \
				isinstance(self.ObservationSpace, spaces.Discrete):

			# one hot encode the state
			proccessed = to_categorical(data, num_classes=self.ObservationSpace.n)


		return proccessed

	def PostProcessSingleColumn(self, data, label):
		proccessed = np.reshape(data, (len(data), -1))

		if (label == DCT.DataColumnTypes.Terminated or
				label == DCT.DataColumnTypes.Truncated):
			# argmax the one hot encoded boolean values
			intBools = np.argmax(data, axis=1)
			proccessed = np.array([bool(i) for i in intBools])

		elif label == DCT.DataColumnTypes.Reward:
			# todo if reward is clipped then we can one hot encode it
			pass

		elif label == DCT.DataColumnTypes.Action:
			if isinstance(self.ActionSpace, spaces.Discrete):
				# argmax the one hot encoded action
				proccessed = np.argmax(data, axis=1)

		elif (label == DCT.DataColumnTypes.CurrentState or
				label == DCT.DataColumnTypes.NextState):
			if isinstance(self.ObservationSpace, spaces.Discrete):
				# argmax the one hot encoded state
				proccessed = np.argmax(data, axis=1)

		return np.array([proccessed])

# endregion
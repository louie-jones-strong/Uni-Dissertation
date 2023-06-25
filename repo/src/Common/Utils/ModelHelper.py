import tensorflow as tf
import redis
import numpy as np
from src.Common.Enums.ModelType import ModelType
import src.Common.Enums.DataColumnTypes as DCT
import typing
from numpy import ndarray as NDArray
from tensorflow.keras.utils import to_categorical
from gymnasium import spaces

RedisClient = redis.Redis(host='model-store', port=5002, db=0)


def BuildModel(modeType:ModelType, observationSpace, actionSpace, config):

	if modeType == ModelType.Policy:
		model = _Build_Policy(observationSpace, actionSpace, config)
	elif modeType == ModelType.Forward:
		model = _Build_Forward(observationSpace, actionSpace, config)



	return model


def FetchNewestWeights(modelType:ModelType, model):

	key = modelType.name

	flatWeightBytes = RedisClient.get(key)
	if flatWeightBytes is None:
		return False

	flatWeights = np.frombuffer(flatWeightBytes, dtype=np.float32)

	currentWeights = model.get_weights()

	newWeights = []
	idx = 0
	for layer in currentWeights:
		shape = layer.shape
		count = np.prod(shape)
		layerWeights = flatWeights[idx:idx+count]
		layerWeights = layerWeights.reshape(shape)
		idx += count
		newWeights.append(layerWeights)

	model.set_weights(newWeights)

	return True

def PushModel(modelType:ModelType, model):

	key = modelType.name

	weights = model.get_weights()
	flatWeights = np.concatenate([w.flatten() for w in weights])

	flatWeightBytes = flatWeights.tobytes()
	RedisClient.set(key, flatWeightBytes)
	return








# region Build Models

def _Build_Policy(observationSpace, actionSpace, config):
	model = tf.keras.models.Sequential()

	# input layer
	model.add(tf.keras.layers.Input(shape=observationSpace))
	model.add(tf.keras.layers.Flatten())

	model.add(tf.keras.layers.Dense(512, activation="relu"))

	# output layer
	model.add(tf.keras.layers.Dense(actionSpace))

	return _FinishModel(model)

def _Build_Forward(observationSpace, actionSpace, config):
	model = tf.keras.models.Sequential()

	# input layer
	model.add(tf.keras.layers.Input(shape=observationSpace))
	model.add(tf.keras.layers.Flatten())
	model.add(tf.keras.layers.Dense(512, activation="relu"))

	# output layer
	model.add(tf.keras.layers.Dense(actionSpace))

	return _FinishModel(model)



def _FinishModel(model):
	# optimizer
	optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

	# compile the network
	model.compile(optimizer=optimizer, loss=tf.keras.losses.Huber(), metrics=["accuracy"])

	return model
# endregion






# region pre and post process columns

	def PreProcessColumns(self,
			columnsData:typing.List[NDArray],
			columnLabels:typing.List[DCT.DataColumnTypes]
			) -> NDArray:

		data = self.PreProcessSingleColumn(columnsData[0], columnLabels[0])

		for i in range(1, len(columnLabels)):
			columnData = self.PreProcessSingleColumn(columnsData[i], columnLabels[i])

			data = np.concatenate((data, columnData), axis=1)

		return data

	def PostProcessColumns(self,
			columnsData:typing.List[NDArray],
			columnLabels:typing.List[DCT.DataColumnTypes]
			) -> typing.List[NDArray]:

		if len(columnLabels) == 1:
			return self.PostProcessSingleColumn(columnsData, columnLabels[0])

		data = []
		for i in range(len(columnLabels)):
			columnData = self.PostProcessSingleColumn(columnsData[i], columnLabels[i])
			data.append(columnData)

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




	def PreProcessSingleColumn(self, data:NDArray, label:DCT.DataColumnTypes) -> NDArray:

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

	def PostProcessSingleColumn(self, data:NDArray, label:DCT.DataColumnTypes) -> NDArray:
		proccessed = np.reshape(data, (len(data), -1))
		proccessed = np.squeeze(proccessed)

		if (label == DCT.DataColumnTypes.Terminated or
				label == DCT.DataColumnTypes.Truncated):
			# argmax the one hot encoded boolean values
			intBools = np.argmax(data, axis=1)
			proccessed = np.array([bool(i) for i in intBools])

		elif label == DCT.DataColumnTypes.Reward:
			# we know that the reward has to be in the rewardRange
			proccessed = np.clip(proccessed, self.RewardRange[0], self.RewardRange[1])

			# todo if reward is clipped then we can one hot encode it

		elif label == DCT.DataColumnTypes.Action and \
				isinstance(self.ActionSpace, spaces.Discrete):

			# argmax the one hot encoded action
			proccessed = np.argmax(data, axis=1)

		elif (label == DCT.DataColumnTypes.CurrentState or
				label == DCT.DataColumnTypes.NextState) and \
				isinstance(self.ObservationSpace, spaces.Discrete):

			# argmax the one hot encoded state
			proccessed = np.argmax(data, axis=1)

		return np.array([proccessed])


	def IsColumnDiscrete(self, label:DCT.DataColumnTypes) -> bool:
		isDiscrete = False

		if (label == DCT.DataColumnTypes.Terminated or
				label == DCT.DataColumnTypes.Truncated):
			isDiscrete = True

		elif label == DCT.DataColumnTypes.Reward:
			# todo if reward is clipped then we can one hot encode it
			# isDiscrete = True
			pass

		elif label == DCT.DataColumnTypes.Action and \
				isinstance(self.ActionSpace, spaces.Discrete):
			isDiscrete = True

		elif (label == DCT.DataColumnTypes.CurrentState or
				label == DCT.DataColumnTypes.NextState) and \
				isinstance(self.ObservationSpace, spaces.Discrete):

			isDiscrete = True


		return isDiscrete
# endregion
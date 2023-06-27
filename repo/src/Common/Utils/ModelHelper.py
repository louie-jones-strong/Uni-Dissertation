import tensorflow as tf
import redis
import numpy as np
from src.Common.Enums.ModelType import ModelType
import src.Common.Enums.DataColumnTypes as DCT
import src.Common.Utils.SharedCoreTypes as SCT
import src.Common.Utils.ConfigHelper as ConfigHelper
import src.Common.Utils.Singleton as Singleton
import typing
from numpy.typing import NDArray
from gymnasium import spaces
from tensorflow.keras.utils import to_categorical


class ModelHelper(Singleton.Singleton):


	def Setup(self, envConfig:SCT.Config):
		self.Config = envConfig

		self.ActionSpace = ConfigHelper.ConfigToSpace(envConfig["ActionSpace"])
		self.ObservationSpace = ConfigHelper.ConfigToSpace(envConfig["ObservationSpace"])
		self.StepRewardRange = envConfig["StepRewardRange"]
		self.EpisodeRewardRange = envConfig["EpisodeRewardRange"]


		# connect to the model store
		self.RedisClient = redis.Redis(host="model-store", port=5002, db=0)
		return

	def BuildModel(self, modeType:ModelType):

		inputColumns = []
		outputColumns = []


		if modeType == ModelType.Forward:
			inputColumns = [DCT.DataColumnTypes.CurrentState, DCT.DataColumnTypes.Action]
			outputColumns = [DCT.DataColumnTypes.NextState,
							DCT.DataColumnTypes.Reward,
							DCT.DataColumnTypes.Terminated]

			model = self._Build_Model(inputColumns, outputColumns)


		return model, inputColumns, outputColumns


	def FetchNewestWeights(self, modelType:ModelType, model):

		key = modelType.name

		flatWeightBytes = self.RedisClient.get(key)
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

	def PushModel(self, modelType:ModelType, model):

		key = modelType.name

		weights = model.get_weights()
		flatWeights = np.concatenate([w.flatten() for w in weights])

		flatWeightBytes = flatWeights.tobytes()
		self.RedisClient.set(key, flatWeightBytes)
		return













# region Build Models
	def _Build_Model(self, inputColumns, outputColumns):

		inputShape = self.GetColumnsShape(inputColumns)



		inputLayer = tf.keras.layers.Input(shape=inputShape)
		flattenLayer = tf.keras.layers.Flatten()(inputLayer)


		# hidden layers
		hiddenLayer = tf.keras.layers.Dense(256, activation="relu")(flattenLayer)
		hiddenLayer = tf.keras.layers.Dense(256, activation="relu")(hiddenLayer)


		# output layers
		outputLayers = []
		losses = {}
		metrics = {}

		for column in outputColumns:

			activation = "linear"
			layerName = column.name

			if self.IsColumnDiscrete(column):
				losses[layerName] = "categorical_crossentropy"
				metrics[layerName] = ["accuracy"]
				activation = "softmax"

			else:
				losses[layerName] = "mse"
				metrics[layerName] = ["mae"]

			outputShape = self.GetColumnsShape([column])
			outputLayer = tf.keras.layers.Dense(outputShape, name=layerName, activation=activation)(hiddenLayer)
			outputLayers.append(outputLayer)

		# create model
		model = tf.keras.models.Model(inputs=inputLayer, outputs=outputLayers)


		# optimizer
		optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

		# compile the network
		model.compile(optimizer=optimizer, loss=losses, metrics=metrics)

		return model
# endregion










# region pre and post process columns

	def GetColumnsShape(self, labels:typing.List[DCT.DataColumnTypes]) -> typing.Tuple[int, ...]:
		shapes = []

		for label in labels:

			if (label == DCT.DataColumnTypes.Terminated or
				label == DCT.DataColumnTypes.Truncated):
				shapes.append((2))


			elif label == DCT.DataColumnTypes.Reward:
				if self.Config["ClipRewards"]:
					shapes.append((3))
				else:
					shapes.append((1))


			elif label == DCT.DataColumnTypes.Action:
				if isinstance(self.ActionSpace, spaces.Discrete):
					# one hot encoded action
					shapes.append((self.ActionSpace.n))
				else:
					shapes.append(self.ActionSpace.shape)


			elif (label == DCT.DataColumnTypes.CurrentState or
					label == DCT.DataColumnTypes.NextState):
				if isinstance(self.ObservationSpace, spaces.Discrete):
					# one hot encoded state
					shapes.append((self.ObservationSpace.n))
				else:
					shapes.append(self.ObservationSpace.shape)

			else:
				raise Exception(f"Unknown column type {label}")



		shape = int(np.sum(np.array(shapes), axis=0))

		return (shape)


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
			if self.Config["ClipRewards"]:
				data = np.sign(data)
				proccessed = to_categorical(data, num_classes=3)

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
			proccessed = np.clip(proccessed, self.StepRewardRange[0], self.StepRewardRange[1])

			# todo if reward is clipped then we can one hot encode it

			if self.Config["ClipRewards"]:
				proccessed = np.argmax(data, axis=1)
				proccessed = proccessed - 1
			else:
				proccessed = np.clip(proccessed, self.StepRewardRange[0], self.StepRewardRange[1])

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
			if self.Config["ClipRewards"]:
				isDiscrete = True

		elif label == DCT.DataColumnTypes.Action and \
				isinstance(self.ActionSpace, spaces.Discrete):
			isDiscrete = True

		elif (label == DCT.DataColumnTypes.CurrentState or
				label == DCT.DataColumnTypes.NextState) and \
				isinstance(self.ObservationSpace, spaces.Discrete):

			isDiscrete = True


		return isDiscrete
# endregion


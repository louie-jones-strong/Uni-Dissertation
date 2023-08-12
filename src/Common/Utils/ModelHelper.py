import tensorflow as tf
import numpy as np
from src.Common.Enums.eModelType import eModelType
import src.Common.Enums.eDataColumnTypes as DCT
import src.Common.Utils.SharedCoreTypes as SCT
import src.Common.Utils.ConfigHelper as ConfigHelper
import src.Common.Utils.Singleton as Singleton
import typing
from typing import Optional
from numpy.typing import NDArray
from gymnasium import spaces
from tensorflow.keras.utils import to_categorical
import src.Common.Store.ModelStore.MsBase as MsBase
import os
from src.Common.Utils.PathHelper import GetRootPath


class ModelHelper(Singleton.Singleton):


	def Setup(self, envConfig:SCT.Config, modelStore:MsBase.MsBase) -> None:
		self.EnvConfig = envConfig

		configPath = os.path.join(GetRootPath(), "Config", "ModelHelper.json")
		self.Config = ConfigHelper.LoadConfig(configPath)

		self.ActionSpace = ConfigHelper.ConfigToSpace(envConfig["ActionSpace"])
		self.ObservationSpace = ConfigHelper.ConfigToSpace(envConfig["ObservationSpace"])
		self.StepRewardRange = envConfig["StepRewardRange"]
		self.EpisodeRewardRange = envConfig["EpisodeRewardRange"]

		self.ModelStore = modelStore
		return

	def BuildModel(self, modeType:eModelType) -> typing.Tuple[
			tf.keras.models.Model,
			typing.List[DCT.eDataColumnTypes],
			typing.List[DCT.eDataColumnTypes],
			SCT.Config]:

		inputColumns = []
		outputColumns = []


		modelConfig = self.Config["ModelConfigs"][modeType.name]

		if modeType == eModelType.Forward:
			inputColumns = [DCT.eDataColumnTypes.CurrentState, DCT.eDataColumnTypes.Action]
			outputColumns = [DCT.eDataColumnTypes.NextState,
							DCT.eDataColumnTypes.Reward,
							DCT.eDataColumnTypes.Terminated]

			model = self._Build_Model(inputColumns, outputColumns, modelConfig)

		elif modeType == eModelType.Value:
			inputColumns = [DCT.eDataColumnTypes.CurrentState]
			outputColumns = [DCT.eDataColumnTypes.MaxFutureRewards]

			model = self._Build_Model(inputColumns, outputColumns, modelConfig)


		elif modeType == eModelType.HumanDiscriminator:
			inputColumns = [DCT.eDataColumnTypes.CurrentState, DCT.eDataColumnTypes.Action]
			outputColumns = [DCT.eDataColumnTypes.PlayStyleTag]
			model = self._Build_Model(inputColumns, outputColumns, modelConfig)

		return model, inputColumns, outputColumns, modelConfig


	def FetchNewestWeights(self, eModelType:eModelType, model:tf.keras.models.Model) -> bool:

		return self.ModelStore.FetchNewestWeights(eModelType.name, model)

	def PushModel(self, eModelType:eModelType, model:tf.keras.models.Model) -> None:

		self.ModelStore.PushModel(eModelType.name, model)

		return













# region Build Models
	def _Build_Model(self,
			inputColumns:typing.List[DCT.eDataColumnTypes],
			outputColumns:typing.List[DCT.eDataColumnTypes],
			modelConfig:SCT.Config) -> tf.keras.models.Model:

		inputShape = self.GetColumnsShape(inputColumns)


		inputLayer = tf.keras.layers.Input(shape=inputShape)
		lastLayer = tf.keras.layers.Flatten()(inputLayer)


		# hidden layers


		def AddLayer(parent, size, activation, dropout, l1, l2):

			layer = tf.keras.layers.Dense(size, activation=activation)(parent)

			if dropout > 0:
				layer = tf.keras.layers.Dropout(rate=dropout)(layer)

			if l1 > 0 or l2 > 0:
				layer = tf.keras.regularizers.L1L2(l1=l1, l2=l2)(layer)

			return layer

		activation = modelConfig["Activations"]
		dropout = modelConfig["Dropout"]
		l1 = modelConfig["L1"]
		l2 = modelConfig["L2"]

		layerSizes = modelConfig["DenseLayers"]
		for layerSize in layerSizes:
			lastLayer = AddLayer(lastLayer, layerSize, activation, dropout, l1, l2)

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
			outputLayer = tf.keras.layers.Dense(outputShape, name=layerName, activation=activation)(lastLayer)
			outputLayers.append(outputLayer)

		# create model
		model = tf.keras.models.Model(inputs=inputLayer, outputs=outputLayers)


		# optimizer
		optimizer = tf.keras.optimizers.Adam(learning_rate=modelConfig["LearningRate"])

		# compile the network
		model.compile(optimizer=optimizer, loss=losses, metrics=metrics)

		return model
# endregion


# region Calculating model metrics

	def CalculateModelMetrics(self, columns, predictions, target, postTarget):
		absErrors = []
		losses = []
		accuracies = []

		for i in range(len(columns)):

			col = columns[i]

			if len(columns) == 1:
				colPredictions = predictions

			else:
				colPredictions = predictions[i]

			metrics = self.CalculateColumnMetrics(col, colPredictions, target[i], postTarget[i])
			absError, loss, accuracy = metrics

			absErrors.append(np.mean(absError, axis=1))
			losses.append(loss)
			accuracies.append(accuracy)
		return absErrors, losses, accuracies

	def CalculateColumnMetrics(self,
			colType:DCT.eDataColumnTypes,
			predicted,
			target,
			postTarget) -> typing.Tuple[float, float, Optional[float]]:

		absErrors = tf.abs(target - predicted)

		lossFunc = self.LossFuncFactory(colType)
		loss = lossFunc(target, predicted)

		accuracy = None
		if self.IsColumnDiscrete(colType):
			accuracyFunc = tf.keras.metrics.CategoricalAccuracy()
			accuracyFunc.reset_state()
			accuracyFunc.update_state(target, predicted)
			accuracy = accuracyFunc.result()

		return absErrors, loss, accuracy

	def LossFuncFactory(self, colType:DCT.eDataColumnTypes) -> tf.keras.losses.Loss:

		if self.IsColumnDiscrete(colType):
			lossFunc = tf.keras.losses.CategoricalCrossentropy()

		else:
			lossFunc = tf.keras.losses.MeanSquaredError()

		return lossFunc

# endregion




# region pre and post process columns

# region get column characteristics

	def GetColumnsShape(self, labels:typing.List[DCT.eDataColumnTypes]) -> typing.Tuple[int, ...]:
		shapes = []

		for label in labels:

			if (label == DCT.eDataColumnTypes.Terminated or
				label == DCT.eDataColumnTypes.Truncated):
				shapes.append([2])


			elif label == DCT.eDataColumnTypes.Reward:
				if self.EnvConfig["ClipRewards"]:
					shapes.append([3])
				else:
					shapes.append([1])

			elif label == DCT.eDataColumnTypes.MaxFutureRewards:
				shapes.append([1])

			elif label == DCT.eDataColumnTypes.Action:
				if isinstance(self.ActionSpace, spaces.Discrete):
					# one hot encoded action
					shapes.append([self.ActionSpace.n])
				else:
					shapes.append(self.ActionSpace.shape)


			elif (label == DCT.eDataColumnTypes.CurrentState or
					label == DCT.eDataColumnTypes.NextState):
				if isinstance(self.ObservationSpace, spaces.Discrete):
					# one hot encoded state
					shapes.append([self.ObservationSpace.n])
				else:
					shapes.append(self.ObservationSpace.shape)

			elif label == DCT.eDataColumnTypes.PlayStyleTag:
				shapes.append([2])

			else:
				raise Exception(f"Unknown column type {label}")



		shape = int(np.sum(np.array(shapes), axis=0))

		return (shape)

	def IsColumnDiscrete(self, label:DCT.eDataColumnTypes) -> bool:
		isDiscrete = False

		if (label == DCT.eDataColumnTypes.Terminated or
				label == DCT.eDataColumnTypes.Truncated):
			isDiscrete = True

		elif label == DCT.eDataColumnTypes.Reward:
			if self.EnvConfig["ClipRewards"]:
				isDiscrete = True

		elif label == DCT.eDataColumnTypes.Action and \
				isinstance(self.ActionSpace, spaces.Discrete):
			isDiscrete = True

		elif (label == DCT.eDataColumnTypes.CurrentState or
				label == DCT.eDataColumnTypes.NextState) and \
				isinstance(self.ObservationSpace, spaces.Discrete):

			isDiscrete = True

		elif label == DCT.eDataColumnTypes.PlayStyleTag:
			isDiscrete = True


		return isDiscrete

# endregion



	def PreProcessColumns(self,
			columnsData:typing.List[NDArray],
			columnLabels:typing.List[DCT.eDataColumnTypes]
			) -> NDArray:

		data = self.PreProcessSingleColumn(columnsData[0], columnLabels[0])

		for i in range(1, len(columnLabels)):
			columnData = self.PreProcessSingleColumn(columnsData[i], columnLabels[i])

			data = np.concatenate((data, columnData), axis=1)

		return data

	def PostProcessColumns(self,
			columnsData:typing.List[NDArray],
			columnLabels:typing.List[DCT.eDataColumnTypes]
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




	def PreProcessSingleColumn(self, data:NDArray, label:DCT.eDataColumnTypes) -> NDArray:
		if len(data) == 0:
			raise Exception(f"Data length is too short for column {label.name}")


		# add a dimension to the data at the end
		proccessed = np.reshape(data, (len(data), -1))

		if label == DCT.eDataColumnTypes.Terminated or label == DCT.eDataColumnTypes.Truncated:
			# one hot encode the boolean values
			intBools = [int(i) for i in data]
			proccessed = to_categorical(intBools, num_classes=2)

		elif label == DCT.eDataColumnTypes.Reward:
			if self.EnvConfig["ClipRewards"]:
				data = np.sign(data)
				proccessed = to_categorical(data+1, num_classes=3)

		elif label == DCT.eDataColumnTypes.Action and \
				isinstance(self.ActionSpace, spaces.Discrete):

			# one hot encode the action
			proccessed = to_categorical(data, num_classes=self.ActionSpace.n)

		elif (label == DCT.eDataColumnTypes.CurrentState or
				label == DCT.eDataColumnTypes.NextState) and \
				isinstance(self.ObservationSpace, spaces.Discrete):

			# one hot encode the state
			proccessed = to_categorical(data, num_classes=self.ObservationSpace.n)


		return proccessed

	def PostProcessSingleColumn(self, data:NDArray, label:DCT.eDataColumnTypes) -> NDArray:
		proccessed = np.reshape(data, (len(data), -1))
		proccessed = np.squeeze(proccessed)

		if (label == DCT.eDataColumnTypes.Terminated or
				label == DCT.eDataColumnTypes.Truncated):
			# argmax the one hot encoded boolean values
			intBools = np.argmax(data, axis=1)
			proccessed = np.array([bool(i) for i in intBools])

		elif label == DCT.eDataColumnTypes.Reward:
			if self.EnvConfig["ClipRewards"]:
				proccessed = np.argmax(data, axis=1)
				proccessed = proccessed - 1
			else:
				proccessed = np.clip(proccessed, self.StepRewardRange[0], self.StepRewardRange[1])

		elif label == DCT.eDataColumnTypes.MaxFutureRewards:
			proccessed = np.clip(proccessed, self.EpisodeRewardRange[0], self.EpisodeRewardRange[1])

			# round the value if clip rewards is on
			if self.EnvConfig["ClipRewards"]:
				proccessed = np.round(proccessed, decimals=0)


		elif label == DCT.eDataColumnTypes.Action and \
				isinstance(self.ActionSpace, spaces.Discrete):

			# argmax the one hot encoded action
			proccessed = np.argmax(data, axis=1)

		elif (label == DCT.eDataColumnTypes.CurrentState or
				label == DCT.eDataColumnTypes.NextState) and \
				isinstance(self.ObservationSpace, spaces.Discrete):

			# argmax the one hot encoded state
			proccessed = np.argmax(data, axis=1)

		return np.array([proccessed])


# endregion


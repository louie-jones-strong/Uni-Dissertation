
import reverb
import src.Common.Enums.ModelType as ModelType
import src.Common.Utils.ModelHelper as ModelHelper
import src.Common.Utils.SharedCoreTypes as SCT
import src.Common.Enums.DataColumnTypes as DCT
import numpy as np
import src.Common.Utils.Metrics.Logger as Logger
import typing
from numpy.typing import NDArray
from gymnasium import spaces
from gymnasium.spaces import Discrete, Box
from tensorflow.keras.utils import to_categorical
from typing import Union


def ConfigToSpace(config:SCT.Config) -> Union[Discrete, Box]:

	if config["Type"] == "Discrete":
		space = Discrete(config["Shape"])
	elif config["Type"] == "Box":
		space = Box(config["Low"], config["High"], config["Shape"], config["Dtype"])

	return space


class Learner:

	def __init__(self, envConfig:SCT.Config, modelType:ModelType):
		self.Config = envConfig
		self.ModelType = modelType


		self.ActionSpace = ConfigToSpace(envConfig["ActionSpace"])
		self.ObservationSpace = ConfigToSpace(envConfig["ObservationSpace"])

		print("build model")
		# todo make this driven by the env config
		self.Model = ModelHelper.BuildModel(self.ModelType, (20,), (16), self.Config)
		print("built model")

		# print("fetching newest weights")
		# didFetch = ModelHelper.FetchNewestWeights(self.ModelType, self.Model)
		# print("fetched newest weights", didFetch)


		self.InputColumns = [DCT.DataColumnTypes.CurrentState, DCT.DataColumnTypes.Action]
		self.OutputColumns = [DCT.DataColumnTypes.NextState]


		self._ConnectToExperienceStore()
		return

	def _ConnectToExperienceStore(self) -> None:

		print("Connecting to experience store")

		self.Store = reverb.TrajectoryDataset.from_table_signature(
			server_address=f'experience-store:{5001}',
			table='Trajectories',
			max_in_flight_samples_per_worker=10)

		# todo should this be configurable?

		print("Connected to experience store")
		return

	def Run(self) -> None:
		print("Starting learner")

		logger = Logger.Logger()
		logCallback = logger.GetFitCallback()

		while True:

			# todo make this configurable
			BatchSize = 32
			ItetationsPerUpdate = 1000 # should this be time based?
			epochs = 5


			batchDataset = self.Store.batch(BatchSize)


			print("Starting training")
			for batch in batchDataset.take(ItetationsPerUpdate):

				raw_x = DCT.FilterDict(self.InputColumns, batch.data)
				x = self.PreProcessColumns(raw_x, self.InputColumns)

				raw_y = DCT.FilterDict(self.OutputColumns, batch.data)
				y = self.PreProcessColumns(raw_y, self.OutputColumns)



				self.Model.fit(x, y, epochs=epochs, callbacks=[logCallback])

			print("Finished training")

			print("Saving model")
			ModelHelper.PushModel(self.ModelType, self.Model)

		return


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


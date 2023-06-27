
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
from tensorflow.keras.utils import to_categorical
import src.Common.Utils.ConfigHelper as ConfigHelper


class Learner:

	def __init__(self, envConfig:SCT.Config, modelType:ModelType, loadModel:bool):
		self.Config = envConfig
		self.ModelType = modelType
		self.ModelHelper = ModelHelper.ModelHelper(envConfig)


		print("build model")
		# todo make this driven by the env config
		self.Model, self.InputColumns, self.OutputColumns = self.ModelHelper.BuildModel(self.ModelType)
		print("built model")

		if loadModel:
			print("fetching newest weights")
			didFetch = self.ModelHelper.FetchNewestWeights(self.ModelType, self.Model)
			print("fetched newest weights", didFetch)

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
			BatchSize = 256
			ItetationsPerUpdate = 1000 # should this be time based?
			epochs = 1


			batchDataset = self.Store.batch(BatchSize)


			print("Starting training")
			for batch in batchDataset.take(ItetationsPerUpdate):

				raw_x = DCT.FilterDict(self.InputColumns, batch.data)
				x = self.ModelHelper.PreProcessColumns(raw_x, self.InputColumns)

				# raw_y = DCT.FilterDict(self.OutputColumns, batch.data)
				# y = self.ModelHelper.PreProcessColumns(raw_y, self.OutputColumns)


				y = []
				for col in self.OutputColumns:
					raw_column = batch.data[col.name]
					column = self.ModelHelper.PreProcessSingleColumn(raw_column, col)
					y.append(column)

				self.Model.fit(x, y, epochs=epochs, callbacks=[logCallback], batch_size=BatchSize)

			print("Finished training")

			print("Saving model")
			self.ModelHelper.PushModel(self.ModelType, self.Model)

		return
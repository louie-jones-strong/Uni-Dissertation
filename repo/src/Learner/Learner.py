
import reverb
import src.Common.Enums.ModelType as ModelType
import src.Common.Utils.ModelHelper as ModelHelper
import src.Common.Utils.SharedCoreTypes as SCT
import src.Common.Enums.DataColumnTypes as DCT
import numpy as np
import src.Common.Utils.Metrics.Logger as Logger

class Learner:

	def __init__(self, envConfig:SCT.Config, modelType:ModelType):
		self.Config = envConfig
		self.ModelType = modelType


		print("build model")
		# todo make this driven by the env config
		self.Model = ModelHelper.BuildModel(self.ModelType, (2,), (1), self.Config)
		print("built model")

		print("fetching newest weights")
		didFetch = ModelHelper.FetchNewestWeights(self.ModelType, self.Model)
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
			BatchSize = 32
			ItetationsPerUpdate = 1000 # should this be time based?


			batchDataset = self.Store.batch(BatchSize)


			print("Starting training")
			for batch in batchDataset.take(ItetationsPerUpdate):
				state = batch.data["State"]
				action = batch.data["Action"]
				x = np.concatenate([state, action], axis=1)

				y = batch.data["NextState"]

				self.Model.fit(x, y, epochs=1, callbacks=[logCallback])

			print("Finished training")

			print("Saving model")
			ModelHelper.PushModel(self.ModelType, self.Model)

		return


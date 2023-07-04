import reverb
import src.Common.Enums.eModelType as eeModelType
import src.Common.Utils.ModelHelper as ModelHelper
import src.Common.Utils.SharedCoreTypes as SCT
import src.Common.Enums.eDataColumnTypes as DCT
import src.Common.Utils.Metrics.Logger as Logger
import time


class Learner:

	def __init__(self, envConfig:SCT.Config, eModelType:eeModelType, loadModel:bool):
		self.Config = envConfig
		self.eModelType = eModelType
		self.ModelHelper = ModelHelper.ModelHelper(envConfig)


		print("build model")
		# todo make this driven by the env config
		self.Model, self.InputColumns, self.OutputColumns = self.ModelHelper.BuildModel(self.eModelType)
		print("built model")

		if loadModel:
			print("fetching newest weights")
			didFetch = self.ModelHelper.FetchNewestWeights(self.eModelType, self.Model)
			print("fetched newest weights", didFetch)

		self._ConnectToExperienceStore()

		self._ModelUpdateTime  = time.time() + self.Config["SecsPerModelPush"]
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

		# todo make this configurable
		BatchSize = 256
		DataCollectionMultiplier = 100
		batchDataset = self.Store.batch(BatchSize * DataCollectionMultiplier)

		while True:

			for batch in batchDataset.take(1):
				# get x data
				raw_x = DCT.FilterDict(self.InputColumns, batch.data)
				x = self.ModelHelper.PreProcessColumns(raw_x, self.InputColumns)

				# get y data
				y = []
				for col in self.OutputColumns:
					raw_column = batch.data[col.name]
					column = self.ModelHelper.PreProcessSingleColumn(raw_column, col)
					y.append(column)


				self.Model.fit(x, y, epochs=1, callbacks=[logCallback], batch_size=BatchSize)

			# should we save the model?
			if time.time() >= self._ModelUpdateTime:
				self._ModelUpdateTime = time.time() + self.Config["SecsPerModelPush"]

				print("Saving model")
				self.ModelHelper.PushModel(self.eModelType, self.Model)

		return
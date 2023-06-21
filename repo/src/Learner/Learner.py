
import src.Common.Utils.ConfigHelper as ConfigHelper
import src.Common.Utils.SharedCoreTypes as SCT
import src.Common.Utils.ModelHelper as ModelHelper
import src.Common.Enums.ModelType as ModelType
import reverb

class Learner(ConfigHelper.ConfigurableClass):

	def __init__(self, envConfig:SCT.Config, modelType:ModelType):
		self.LoadConfig(envConfig)

		self.ModelType = modelType


		self._SetupModel()
		self._ConnectToExperienceStore()
		return

	def _SetupModel(self) -> None:
		print("build model")
		self.Model = ModelHelper.BuildModel(self.ModelType, (1,), (1), {}) # todo make this driven by the env config
		print("built model")

		print("fetching newest weights")
		didFetch = ModelHelper.FetchNewestWeights(self.ModelType, self.Model)
		print("fetched newest weights", didFetch)
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

		while True:

			# todo make this configurable
			BatchSize = 32
			ItetationsPerUpdate = 1000 # should this be time based?


			batchDataset = self.Store.batch(BatchSize)


			print("Starting training")
			for batch in batchDataset.take(ItetationsPerUpdate):
				x = batch.data["State"]
				y = batch.data["Action"]
				print(x.shape)
				self.Model.fit(x, y, epochs=1)

			print("Finished training")

			print("Saving model")
			ModelHelper.PushModel(self.ModelType, self.Model)

		return


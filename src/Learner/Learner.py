import reverb
from src.Common.Enums.eModelType import eModelType
import src.Common.Utils.ModelHelper as ModelHelper
import src.Common.Enums.eDataColumnTypes as DCT
import src.Common.Utils.Metrics.Logger as Logger
import time
import tensorflow as tf
import src.Common.Store.ExperienceStore.EsReverb as EsReverb
import src.Common.Store.ExperienceStore.EsNumpy as EsNumpy
import numpy as np
from tensorflow.keras.utils import to_categorical
from src.Common.Utils.Config.ConfigurableClass import ConfigurableClass
import os

class Learner(ConfigurableClass):

	def __init__(self, modelType:eModelType, loadModel:bool, envDataPath:str):
		self.LoadConfig()
		self.ModelType = modelType
		self.EnvDataPath = envDataPath

		self.ModelHelper = ModelHelper.ModelHelper(self.EnvConfig)

		print("build model")
		modelData = self.ModelHelper.BuildModel(self.ModelType)
		self.Model, self.InputColumns, self.OutputColumns, self.ModelConfig = modelData

		self.UsePriorities = self.ModelType != eModelType.PlayStyle_Discriminator \
						and self.ModelType != eModelType.Human_Discriminator



		self.HumanData = None
		if self.ModelType == eModelType.Human_Discriminator or self.ModelType == eModelType.PlayStyle_Discriminator:

			examplePath = os.path.join(self.EnvDataPath, "examples", self.ModelConfig["ReplayExamples"])

			self.HumanData = EsNumpy.EsNumpy(examplePath)
			self.HumanData.Load()

		print("built model")

		if loadModel:
			print("fetching newest weights")
			didFetch = self.ModelHelper.FetchNewestWeights(self.ModelType, self.Model)
			print("fetched newest weights", didFetch)

		self._ConnectToExperienceStore()

		self._ModelUpdateTime = time.time() + self.EnvConfig["SecsPerModelPush"]
		self._Logger = Logger.Logger()
		return

	def _ConnectToExperienceStore(self) -> None:

		maxInFlightSamples = self.Config["LearnerConfig"]["MaxInFlightSamples"]
		batchSize = self.ModelConfig["TrainingBatchSize"]
		dataCollectionMultiplier = self.Config["LearnerConfig"]["DataCollectionMultiplier"]

		# connect to the experience store dataset
		trajectoryDataset = reverb.TrajectoryDataset.from_table_signature(
			server_address=f'experience-store:{5001}',
			table=self.ModelConfig["DataTable"],
			max_in_flight_samples_per_worker=maxInFlightSamples)

		self.BatchedTrajectoryDataset = trajectoryDataset.batch(batchSize * dataCollectionMultiplier)


		self.EsStore = EsReverb.EsReverb()

		return


	def Run(self) -> None:
		print("Starting learner")

		while True:

			x, y, post_y, batchInfo = self._GetBatch()

			if self.UsePriorities:  # using priorities

				self._Logger.LogDict({"in_priority": batchInfo.priority})

				absErrors = self._TuneModelGradTape(x, y, post_y)

				self._Logger.LogDict({"out_priority": absErrors})
				self.EsStore.UpdatePriorities(self.ModelConfig["DataTable"], batchInfo.key, absErrors)

			else:  # not using priorities
				self._TuneModelFit(x, y, post_y)



			# should we save the model?
			if time.time() >= self._ModelUpdateTime:
				self._ModelUpdateTime = time.time() + self.EnvConfig["SecsPerModelPush"]

				print("Saving model")
				self.ModelHelper.PushModel(self.ModelType, self.Model)

		return


	def _GetBatch(self):

		batch = self.BatchedTrajectoryDataset.take(1).__iter__().__next__()

		# get x data
		raw_x = DCT.FilterDict(self.InputColumns, batch.data)
		x = self.ModelHelper.PreProcessColumns(raw_x, self.InputColumns)

		if self.ModelType == eModelType.PlayStyle_Discriminator or self.ModelType == eModelType.Human_Discriminator:

			# add the human data concatenated to the end of the batch

			humanDataColumns = [self.HumanData.States,
				self.HumanData.NextStates,
				self.HumanData.Actions,
				self.HumanData.Rewards,
				self.HumanData.FutureRewards,
				self.HumanData.Terminated,
				self.HumanData.Truncated]


			human_raw_x = DCT.FilterColumns(self.InputColumns, humanDataColumns)
			human_x = self.ModelHelper.PreProcessColumns(human_raw_x, self.InputColumns)
			human_y = np.ones((len(human_x), 1))

			post_y = np.zeros((len(x), 1))

			x = np.concatenate((x, human_x), axis=0)
			post_y = np.concatenate((post_y, human_y), axis=0)

			y = to_categorical(post_y, num_classes=2)




		else:

			# get y data
			y = []
			post_y = []
			for col in self.OutputColumns:
				raw_column = batch.data[col.name]
				column = self.ModelHelper.PreProcessSingleColumn(raw_column, col)
				y.append(column)
				post_y.append(raw_column)


		batchInfo = batch.info
		return x, y, post_y, batchInfo


	def _TuneModelFit(self, x, y, post_y) -> None:
		history = self.Model.fit(x, y, epochs=1, batch_size=self.ModelConfig["TrainingBatchSize"])

		accuracy = history.history["accuracy"][0]
		loss = history.history["loss"][0]

		col = self.OutputColumns[0]
		self._Logger.LogDict({
				f"Train_{col.name}_accuracy": accuracy,
				f"Train_{col.name}_loss": loss
			})
		return

	def _TuneModelGradTape(self, x, y, post_y) -> None:


		with tf.GradientTape() as tape:
			predictions = self.Model(x)

			metrics = self.ModelHelper.CalculateModelMetrics(self.OutputColumns, predictions, y, post_y)
			absErrors, losses, accuracies = metrics


		# update the model
		gradients = tape.gradient(losses, self.Model.trainable_variables)
		self.Model.optimizer.apply_gradients(zip(gradients, self.Model.trainable_variables))


		# log the metrics
		logDict = {}
		for i in range(len(self.OutputColumns)):
			col = self.OutputColumns[i]
			loss = losses[i]
			accuracy = accuracies[i]

			logDict[f"Train_{col.name}_Loss"] = loss

			if accuracy is not None:
				logDict[f"Train_{col.name}_Accuracy"] = accuracy

		self._Logger.LogDict(logDict)


		# calculate priorities
		absErrors = np.array(absErrors)
		absErrors = np.max(absErrors, axis=0)

		return absErrors
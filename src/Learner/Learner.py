import reverb
from src.Common.Enums.eModelType import eModelType
import src.Common.Utils.ModelHelper as ModelHelper
import src.Common.Utils.SharedCoreTypes as SCT
import src.Common.Enums.eDataColumnTypes as DCT
import src.Common.Utils.Metrics.Logger as Logger
import time
import tensorflow as tf
import src.Common.Store.ExperienceStore.EsReverb as EsReverb
import src.Common.Store.ExperienceStore.EsNumpy as EsNumpy
import numpy as np
from tensorflow.keras.utils import to_categorical

class Learner:

	def __init__(self, envConfig:SCT.Config, modelType:eModelType, loadModel:bool, runPath:str):
		self.Config = envConfig
		self.ModelType = modelType
		self.ModelHelper = ModelHelper.ModelHelper(envConfig)
		self.RunPath = runPath

		print("build model")
		# todo make this driven by the env config
		self.Model, self.InputColumns, self.OutputColumns, self.DataTable = self.ModelHelper.BuildModel(self.ModelType)
		self.BatchSize = 256

		self.UsePriorities = self.ModelType != eModelType.HumanDiscriminator

		self.HumanData = None
		if self.ModelType == eModelType.HumanDiscriminator:
			self.HumanData = EsNumpy.EsNumpy(self.RunPath)
			self.HumanData.Load()

		print("built model")

		if loadModel:
			print("fetching newest weights")
			didFetch = self.ModelHelper.FetchNewestWeights(self.ModelType, self.Model)
			print("fetched newest weights", didFetch)

		self._ConnectToExperienceStore()

		self._ModelUpdateTime = time.time() + self.Config["SecsPerModelPush"]
		self._Logger = Logger.Logger()
		return

	def _ConnectToExperienceStore(self) -> None:
		dataCollectionMultiplier = 1

		# connect to the experience store dataset
		trajectoryDataset = reverb.TrajectoryDataset.from_table_signature(
			server_address=f'experience-store:{5001}',
			table=self.DataTable,
			max_in_flight_samples_per_worker=3)


		self.BatchedTrajectoryDataset = trajectoryDataset.batch(self.BatchSize * dataCollectionMultiplier)


		self.EsStore = EsReverb.EsReverb(self.RunPath)

		return


	def Run(self) -> None:
		print("Starting learner")

		while True:

			x, y, post_y, batchInfo = self._GetBatch()

			if self.UsePriorities:  # using priorities

				self._Logger.LogDict({"in_priority": batchInfo.priority})

				absErrors = self._TuneModelGradTape(x, y, post_y)

				self._Logger.LogDict({"out_priority": absErrors})
				self.EsStore.UpdatePriorities(self.DataTable, batchInfo.key, absErrors)

			else:  # not using priorities
				self._TuneModelFit(x, y, post_y)



			# should we save the model?
			if time.time() >= self._ModelUpdateTime:
				self._ModelUpdateTime = time.time() + self.Config["SecsPerModelPush"]

				print("Saving model")
				self.ModelHelper.PushModel(self.ModelType, self.Model)

		return


	def _GetBatch(self):

		batch = self.BatchedTrajectoryDataset.take(1).__iter__().__next__()

		# get x data
		raw_x = DCT.FilterDict(self.InputColumns, batch.data)
		x = self.ModelHelper.PreProcessColumns(raw_x, self.InputColumns)

		if self.ModelType == eModelType.HumanDiscriminator:

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
		history = self.Model.fit(x, y, epochs=1, batch_size=self.BatchSize)

		accuracy = history.history["accuracy"][0]
		loss = history.history["loss"][0]

		col = self.OutputColumns[0]
		self._Logger.LogDict({
				f"{col.name}_accuracy": accuracy,
				f"{col.name}_loss": loss
			})
		return

	def _TuneModelGradTape(self, x, y, post_y) -> None:

		losses = []
		logDict = {}
		absErrors = []

		accuracyCal = tf.keras.metrics.Accuracy()

		with tf.GradientTape() as tape:
			predictions = self.Model(x)

			# loop through each column and calculate the loss
			for i in range(len(self.OutputColumns)):

				col = self.OutputColumns[i]
				colY = y[i]
				colPost_y = post_y[i]

				if len(self.OutputColumns) == 1:
					colPredictions = predictions

				else:
					colPredictions = predictions[i]

				lossFunc = tf.keras.losses.MeanSquaredError()

				absError = tf.abs(colY - colPredictions)
				loss = lossFunc(colY, colPredictions)

				losses.append(loss)
				absErrors.append(np.mean(absError, axis=1))



				logDict[f"{col.name}_loss"] = loss.numpy()

				if self.ModelHelper.IsColumnDiscrete(col):
					# reshape the predictions to match the post processed y
					postPredictions = self.ModelHelper.PostProcessSingleColumn(colPredictions, col)
					postPredictions = tf.reshape(postPredictions, colPost_y.shape)

					accuracyCal.reset_state()
					accuracyCal.update_state(colPost_y, postPredictions)
					accuracy = accuracyCal.result()
					logDict[f"{col.name}_accuracy"] = accuracy.numpy()

		absErrors = np.array(absErrors)
		absErrors = np.max(absErrors, axis=0)

		gradients = tape.gradient(losses, self.Model.trainable_variables)

		self.Model.optimizer.apply_gradients(zip(gradients, self.Model.trainable_variables))

		self._Logger.LogDict(logDict)


		return absErrors
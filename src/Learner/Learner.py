import reverb
import src.Common.Enums.eModelType as eeModelType
import src.Common.Utils.ModelHelper as ModelHelper
import src.Common.Utils.SharedCoreTypes as SCT
import src.Common.Enums.eDataColumnTypes as DCT
import src.Common.Utils.Metrics.Logger as Logger
import time
import tensorflow as tf
import src.Common.Store.ExperienceStore.EsReverb as EsReverb
import numpy as np


class Learner:

	def __init__(self, envConfig:SCT.Config, eModelType:eeModelType, loadModel:bool):
		self.Config = envConfig
		self.eModelType = eModelType
		self.ModelHelper = ModelHelper.ModelHelper(envConfig)


		print("build model")
		# todo make this driven by the env config
		self.Model, self.InputColumns, self.OutputColumns, self.DataTable = self.ModelHelper.BuildModel(self.eModelType)
		self.BatchSize = 256

		print("built model")

		if loadModel:
			print("fetching newest weights")
			didFetch = self.ModelHelper.FetchNewestWeights(self.eModelType, self.Model)
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
			max_in_flight_samples_per_worker=10)


		self.BatchedTrajectoryDataset = trajectoryDataset.batch(self.BatchSize * dataCollectionMultiplier)


		self.EsStore = EsReverb.EsReverb()

		return


	def Run(self) -> None:
		print("Starting learner")

		while True:

			for batch in self.BatchedTrajectoryDataset.take(1):
				# get x data
				raw_x = DCT.FilterDict(self.InputColumns, batch.data)
				x = self.ModelHelper.PreProcessColumns(raw_x, self.InputColumns)

				# get y data
				y = []
				post_y = []
				for col in self.OutputColumns:
					raw_column = batch.data[col.name]
					column = self.ModelHelper.PreProcessSingleColumn(raw_column, col)
					y.append(column)
					post_y.append(raw_column)


				absErrors = self._TuneModelGradTape(x, y, post_y)

				self._Logger.LogDict({"in_priority": batch.info.priority})
				self._Logger.LogDict({"out_priority": absErrors})

				self.EsStore.UpdatePriorities(self.DataTable, batch.info.key, absErrors)




			# should we save the model?
			if time.time() >= self._ModelUpdateTime:
				self._ModelUpdateTime = time.time() + self.Config["SecsPerModelPush"]

				print("Saving model")
				self.ModelHelper.PushModel(self.eModelType, self.Model)

		return



	def _TuneModelFit(self, x, y, post_y) -> None:

		logger = Logger.Logger()
		tuneCallbacks = []
		tuneCallbacks.append(logger.GetFitCallback())

		self.Model.fit(x, y, epochs=1, callbacks=tuneCallbacks, batch_size=self.BatchSize)
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
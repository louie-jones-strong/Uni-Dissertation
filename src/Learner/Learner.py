import reverb
import src.Common.Enums.eModelType as eeModelType
import src.Common.Utils.ModelHelper as ModelHelper
import src.Common.Utils.SharedCoreTypes as SCT
import src.Common.Enums.eDataColumnTypes as DCT
import src.Common.Utils.Metrics.Logger as Logger
import time
import tensorflow as tf


class Learner:

	def __init__(self, envConfig:SCT.Config, eModelType:eeModelType, loadModel:bool):
		self.Config = envConfig
		self.eModelType = eModelType
		self.ModelHelper = ModelHelper.ModelHelper(envConfig)


		print("build model")
		# todo make this driven by the env config
		self.Model, self.InputColumns, self.OutputColumns = self.ModelHelper.BuildModel(self.eModelType)
		self.BatchSize = 256

		print("built model")

		if loadModel:
			print("fetching newest weights")
			didFetch = self.ModelHelper.FetchNewestWeights(self.eModelType, self.Model)
			print("fetched newest weights", didFetch)

		self._ConnectToExperienceStore()

		self._ModelUpdateTime  = time.time() + self.Config["SecsPerModelPush"]
		return

	def _ConnectToExperienceStore(self) -> None:
		dataCollectionMultiplier = 1

		# connect to the experience store dataset
		trajectoryDataset = reverb.TrajectoryDataset.from_table_signature(
			server_address=f'experience-store:{5001}',
			table='Trajectories',
			max_in_flight_samples_per_worker=10)


		self.BatchedTrajectoryDataset = trajectoryDataset.batch(self.BatchSize * dataCollectionMultiplier)


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

				self._TuneModelGradTape(x, y, post_y)



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

		accuracyCal = tf.keras.metrics.Accuracy()

		with tf.GradientTape() as tape:
			predictions = self.Model(x)

			# loop through each column and calculate the loss
			for i in range(len(self.OutputColumns)):

				print("calculating loss for column", i)
				col = self.OutputColumns[i]
				colPredictions = predictions[i]

				colY = y[i]
				colPost_y = post_y[i]
				lossFunc = tf.keras.losses.MeanSquaredError()

				absError = tf.abs(colY - colPredictions)
				loss = lossFunc(colY, colPredictions)

				# loss = tf.reduce_mean(loss * importance)
				losses.append(loss)

				# reshape the predictions to match the post processed y
				postPredictions = self.ModelHelper.PostProcessSingleColumn(colPredictions, col)

				postPredictions = tf.reshape(postPredictions, colPost_y.shape)
				accuracyCal.reset_state()
				accuracyCal.update_state(colPost_y, postPredictions)
				accuracy = accuracyCal.result()

				logDict[f"{col.name}_loss"] = loss.numpy()
				logDict[f"{col.name}_absError"] = absError.numpy()
				logDict[f"{col.name}_accuracy"] = accuracy.numpy()



		gradients = tape.gradient(losses, self.Model.trainable_variables)

		self.Model.optimizer.apply_gradients(zip(gradients, self.Model.trainable_variables))


		logger = Logger.Logger()
		logger.LogDict(logDict)

		return
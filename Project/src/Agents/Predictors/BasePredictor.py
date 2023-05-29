import src.DataManager.DataManager as DataManager
import src.Utils.Metrics.Logger as Logger
import src.DataManager.DataColumnTypes as DCT
import typing
import src.Utils.SharedCoreTypes as SCT
import numpy as np
from numpy.typing import NDArray
from collections import deque


class BasePredictor:

	def __init__(self,
			xLabels:typing.List[DCT.DataColumnTypes],
			yLabels:typing.List[DCT.DataColumnTypes]):
		self.LoadConfig({})
		assert len(xLabels) > 0, "xLabels must have at least one element"
		assert len(yLabels) > 0, "yLabels must have at least one element"

		self._DataManager = DataManager.DataManager()
		self._Logger = Logger.Logger()

		self._XLabels = xLabels
		self._YLabels = yLabels
		self._IsDiscrete = all([self._DataManager.IsColumnDiscrete(y) for y in yLabels])

		className = self.__class__.__name__.replace("Predictor", "")
		predictionName = "".join([y.name for y in yLabels])
		predictionName = predictionName.replace("DataColumnTypes.", "")
		self._Name = f"{predictionName}_{className}"

		self._StepsSinceTrained = -1
		self._ErrorQueue = deque()
		self._AccuracyQueue = deque()
		return

	def LoadConfig(self, config:SCT.Config) -> None:
		self._Config = config
		self._Config["MinStepsBetweenTraining"] = 100
		self._Config["MinAccuracy"] = 0.99
		self._Config["MaxError"] = 0.1
		self._Config["ValidationAverageWindow"] = 100
		return

	def Save(self, folderPath:str) -> None:
		return

	def Load(self, folderPath:str) -> None:
		return

	@staticmethod
	def _ItemWiseEqual(prediction:NDArray, target:NDArray):

		itemsEqual = prediction == target
		itemsEqual = np.squeeze(itemsEqual)

		if len(itemsEqual.shape) > 1:
			itemsEqual = np.all(itemsEqual, axis=1)

		return itemsEqual


	def Predict(self, x:typing.List[NDArray]) -> typing.Tuple[NDArray, NDArray[np.float32]]:

		if self._StepsSinceTrained < 0:
			return None, 0.0


		proccessedX = self._DataManager.PreProcessColumns(x, self._XLabels)
		rawPrediction = self.CalRawPrediction(proccessedX)

		proccessedPrediction = self._DataManager.PostProcessColumns(rawPrediction, self._YLabels)

		# todo: confidence
		confidence = np.ones(len(x), dtype=np.float32) * 0.5
		return proccessedPrediction, confidence

	def CalRawPrediction(self, proccessedX):
		if self._StepsSinceTrained < 0:
			return None, 0.0

		rawPrediction = self._Predict(proccessedX)
		return rawPrediction

	def Observe(self, x, y) -> None:

		# train the model if we haven't trained yet
		if self._StepsSinceTrained < 0:
			self.Train()

		else:
			# validate the model, if we have a trained model
			proccessedY = self._DataManager.PreProcessColumns(y, self._YLabels)

			proccessedX = self._DataManager.PreProcessColumns(x, self._XLabels)
			predicted = self._Predict(proccessedX)

			error, accuracy = self._Evaluate(predicted, proccessedY, y)


			self._ErrorQueue.append(error)
			if len(self._ErrorQueue) > self._Config["ValidationAverageWindow"]:
				self._ErrorQueue.popleft()

			self._AccuracyQueue.append(accuracy)
			if len(self._AccuracyQueue) > self._Config["ValidationAverageWindow"]:
				self._AccuracyQueue.popleft()

			self._Logger.LogDict({
				f"{self._Name}_Validation_Error": error,
				f"{self._Name}_Validation_Accuracy": accuracy
			})
			self._StepsSinceTrained += 1

			self.Train()
		return

	def Train(self) -> bool:

		# check if we have trained recently
		if self._StepsSinceTrained >= 0 and \
				self._StepsSinceTrained < self._Config["MinStepsBetweenTraining"]:
			return False

		avgError = np.mean(self._ErrorQueue)
		avgAccuracy = np.mean(self._AccuracyQueue)

		modelInBounds = avgError < self._Config["MaxError"] and \
			((not self._IsDiscrete) or avgAccuracy > self._Config["MinAccuracy"])

		if modelInBounds and self._StepsSinceTrained >= 0:
			return False

		x, y = self._DataManager.GetXYData(self._XLabels, self._YLabels)

		# check if we have enough data to train
		if len(y) == 0 or len(y[0]) < 2:
			return False

		proccessedX = self._DataManager.PreProcessColumns(x, self._XLabels)
		proccessedY = self._DataManager.PreProcessColumns(y, self._YLabels)

		wasTrained = self._Train(proccessedX, proccessedY)

		# evaluate the model if we trained it
		if wasTrained:
			self._ErrorQueue.clear()
			self._AccuracyQueue.clear()

			self._StepsSinceTrained = 0

			prediction = self._Predict(proccessedX)
			error, accuracy = self._Evaluate(prediction, proccessedY, y)

			self._Logger.LogDict({
				f"{self._Name}_Trained_Error": error,
				f"{self._Name}_Trained_Accuracy": accuracy
			})

		return wasTrained



	def _Evaluate(self, rawPrediction, proccessedY, y):

		error = abs(np.mean(proccessedY - rawPrediction))

		prediction = self._DataManager.PostProcessColumns(rawPrediction, self._YLabels)


		accuracy = np.mean(self._ItemWiseEqual(prediction, y))

		return error, accuracy

	def _Predict(self, x:NDArray) -> NDArray:
		return None

	def _Train(self, x:NDArray, y:NDArray) -> bool:
		return False






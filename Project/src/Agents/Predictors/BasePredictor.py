import src.DataManager.DataManager as DataManager
import src.Utils.Metrics.Logger as Logger
import src.DataManager.DataColumnTypes as DCT
import typing
import src.Utils.SharedCoreTypes as SCT
import numpy as np
from numpy.typing import NDArray


class BasePredictor:

	def __init__(self,
			xLabels:typing.List[DCT.DataColumnTypes],
			yLabels:typing.List[DCT.DataColumnTypes]):

		assert len(xLabels) > 0, "xLabels must have at least one element"
		assert len(yLabels) > 0, "yLabels must have at least one element"

		self._XLabels = xLabels
		self._YLabels = yLabels

		className = self.__class__.__name__.replace("Predictor", "")
		predictionName = "".join([y.name for y in yLabels])
		predictionName = predictionName.replace("DataColumnTypes.", "")
		self._Name = f"{predictionName}_{className}"


		self._DataManager = DataManager.DataManager()
		self._Logger = Logger.Logger()

		self._FramesSinceTrained = -1

		return

	def LoadConfig(self, config:SCT.Config) -> None:
		return

	def Save(self, folderPath:str) -> None:
		return

	def Load(self, folderPath:str) -> None:
		return





	def Predict(self, x:NDArray) -> typing.Tuple[NDArray, NDArray[np.float32]]:

		if self._FramesSinceTrained < 0:
			return None, 0.0

		proccessedX = self._DataManager.PreProcessColumns(x, self._XLabels)

		predicted = self._Predict(proccessedX)
		proccessedPrediction = self._DataManager.PostProcessColumns(predicted, self._YLabels)

		# todo: confidence
		confidence = np.ones(len(x), dtype=np.float32) * 0.5
		return proccessedPrediction, confidence

	def Observe(self, x, y) -> None:

		if self._FramesSinceTrained >= 0:

			proccessedY = self._DataManager.PreProcessColumns(y, self._YLabels)

			proccessedX = self._DataManager.PreProcessColumns(x, self._XLabels)
			predicted = self._Predict(proccessedX)

			error = self._Evaluate(predicted, proccessedY)
			self._Logger.LogDict({
				f"{self._Name}_Validation_Error": error
			})

			self._FramesSinceTrained += 1

		# todo: do we need to train this frame?
		if self._FramesSinceTrained >= 10 or self._FramesSinceTrained < 0:
			self.Train()
		return

	def Train(self) -> None:
		x, y = self._DataManager.GetXYData(self._XLabels, self._YLabels)

		if len(y) == 0 or len(y[0]) < 2:
			return

		proccessedX = self._DataManager.PreProcessColumns(x, self._XLabels)
		proccessedY = self._DataManager.PreProcessColumns(y, self._YLabels)

		self._Train(proccessedX, proccessedY)
		self._FramesSinceTrained = 0

		# evaluate the model
		prediction = self._Predict(proccessedX)
		error = self._Evaluate(prediction, proccessedY)

		self._Logger.LogDict({
			f"{self._Name}_Trained_Error": error
		})
		return

	def _Evaluate(self, rawPrediction, proccessedY):

		error = abs(np.mean(proccessedY - rawPrediction))
		return error

	def _Predict(self, x:NDArray) -> NDArray:
		if self._FramesSinceTrained < 0:
			self.Train()

		predicted = None
		return predicted

	def _Train(self, x:NDArray, y:NDArray) -> None:
		return
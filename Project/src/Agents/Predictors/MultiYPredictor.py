from src.Agents.Predictors import BasePredictor, EnsemblePredictor
import numpy as np
from numpy.typing import NDArray


class MultiYPredictor(BasePredictor.BasePredictor):

	def __init__(self, xLabels, yLabels):
		super().__init__(xLabels, yLabels)
		self._Predictors = {}
		for yLabel in yLabels:
			self._Predictors[yLabel] = EnsemblePredictor.EnsemblePredictor(xLabels, [yLabel])

		return

	def Observe(self, x, y):
		super().Observe(x, y)
		assert len(y) == len(self._YLabels), "y must have the same number of elements as yLabels"

		for i in range(len(self._YLabels)):
			yLabel = self._YLabels[i]
			self._Predictors[yLabel].Observe(x, y[i])

		return

	def _Evaluate(self, rawPrediction, proccessedY):

		data = rawPrediction[0]
		for i in range(1, len(rawPrediction)):
			data = np.concatenate((data, rawPrediction[i]), axis=1)

		return super()._Evaluate(data, proccessedY)

	def _Predict(self, x:NDArray) -> NDArray:
		super()._Predict(x)

		predicted = []
		for yLabel in self._YLabels:
			prediction = self._Predictors[yLabel]._Predict(x)
			predicted.append(prediction)

		return np.array(predicted)

	def _Train(self, x:NDArray, y:NDArray) -> None:
		super()._Train(x, y)

		for yLabel in self._YLabels:
			self._Predictors[yLabel].Train()
		return


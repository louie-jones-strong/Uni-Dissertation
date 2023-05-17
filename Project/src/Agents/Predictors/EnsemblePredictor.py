from . import BasePredictor, DecisonTreePredictor, LinearPredictor
import numpy as np
from numpy.typing import NDArray

class EnsemblePredictor(BasePredictor.BasePredictor):

	def __init__(self, xLabels, yLabels):
		super().__init__(xLabels, yLabels)

		self._Predictors = [
			DecisonTreePredictor.DecisonTreePredictor(xLabels, yLabels),
			LinearPredictor.LinearPredictor(xLabels, yLabels)
		]
		return


	def Observe(self, x, y) -> None:
		super().Observe(x, y)

		for predictor in self._Predictors:
			predictor.Observe(x, y)

		return


	def _Predict(self, proccessedX:NDArray) -> NDArray:
		super()._Predict(proccessedX)

		predictions = []

		for i in range(len(self._Predictors)):
			prediction = self._Predictors[i]._Predict(proccessedX)
			predictions.append(prediction)

		predictions = np.array(predictions)

		predictions = np.mean(predictions, axis=0)
		return predictions
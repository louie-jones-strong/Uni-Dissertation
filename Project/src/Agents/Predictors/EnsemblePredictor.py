from . import BasePredictor, DecisonTreePredictor, LinearRegressionPredictor
import numpy as np

class EnsemblePredictor(BasePredictor.BasePredictor):

	def __init__(self, xLabels, yLabels, name):
		super().__init__(xLabels, yLabels, name)

		self._Predictors = [
			DecisonTreePredictor.DecisonTreePredictor(xLabels, yLabels, name),
			LinearRegressionPredictor.LinearRegressionPredictor(xLabels, yLabels, name)
		]
		return

	def _Predict(self, x):
		super()._Predict(x)

		predicted = self._Predictors[0].Predict(x).astype(np.float32)
		for i in range(1, len(self._Predictors)):
			predicted += self._Predictors[i].Predict(x).astype(np.float32)

		return predicted / len(self._Predictors)

	def _Train(self, x, y):
		super()._Train(x, y)

		for predictor in self._Predictors:
			predictor.Train()

		return


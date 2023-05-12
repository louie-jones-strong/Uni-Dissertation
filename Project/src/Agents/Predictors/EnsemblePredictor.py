from . import BasePredictor, DecisonTreePredictor, LinearRegressionPredictor
import numpy as np

class EnsemblePredictor(BasePredictor.BasePredictor):

	def __init__(self, xLabels, yLabels):
		super().__init__(xLabels, yLabels)

		self._Predictors = [
			DecisonTreePredictor.DecisonTreePredictor(xLabels, yLabels),
			# LinearRegressionPredictor.LinearRegressionPredictor(xLabels, yLabels)
		]
		return

	def Predict(self, x):
		super().Predict(x)

		predicted = self._Predictors[0].Predict(x).astype(np.float32)
		for i in range(1, len(self._Predictors)):
			predicted += self._Predictors[i].Predict(x).astype(np.float32)

		return predicted / len(self._Predictors)

	def Train(self):
		super().Train()

		for predictor in self._Predictors:
			predictor.Train()

		self._Evaluate()
		return


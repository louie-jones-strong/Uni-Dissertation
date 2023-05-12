from src.Agents.Predictors import BasePredictor, EnsemblePredictor
import numpy as np


class MultiYPredictor(BasePredictor.BasePredictor):

	def __init__(self, xLabels, yLabels):
		super().__init__(xLabels, yLabels)
		self._Predictors = {}
		for yLabel in yLabels:
			self._Predictors[yLabel] = EnsemblePredictor.EnsemblePredictor(xLabels, [yLabel])

		return

	def Observe(self, x, y):
		super().Observe(x, y)

		for i in range(len(self._YLabels)):
			yLabel = self._YLabels[i]
			self._Predictors[yLabel].Observe(x, y[i])

		return

	def Predict(self, x):
		super().Predict(x)

		predicted = []
		for yLabel in self._YLabels:
			predicted.append(self._Predictors[yLabel].Predict(x))

		return predicted

	def Train(self):
		super().Train()

		for yLabel in self._YLabels:
			self._Predictors[yLabel].Train()


		self._Evaluate()
		return


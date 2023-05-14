from . import BasePredictor
import numpy as np
from sklearn.linear_model import LinearRegression

class LinearRegressionPredictor(BasePredictor.BasePredictor):

	def __init__(self, xLabels, yLabels, name):
		super().__init__(xLabels, yLabels, name)
		self.Predictor = LinearRegression()
		return

	def _Predict(self, x):
		super()._Predict(x)

		predicted = self.Predictor.predict(x)

		return predicted

	def _Train(self, x, y):
		super()._Train(x, y)

		self.Predictor.fit(x, y)
		return


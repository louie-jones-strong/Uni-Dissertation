from . import BasePredictor
import numpy as np
from sklearn.linear_model import LinearRegression

class LinearRegressionPredictor(BasePredictor.BasePredictor):

	def __init__(self, xLabels, yLabels):
		super().__init__(xLabels, yLabels)
		self.Predictor = LinearRegression()
		return

	def Predict(self, x):
		super().Predict(x)

		predicted = self.Predictor.predict(x)

		return predicted

	def Train(self):
		super().Train()

		x, y = self.DataManager.GetXYData(self._XLabels, self._YLabels)

		self.Predictor.fit(x, y)

		self._Evaluate()
		return


from . import BasePredictor
from sklearn.linear_model import LinearRegression
from numpy.typing import NDArray

class LinearPredictor(BasePredictor.BasePredictor):

	def __init__(self, xLabels, yLabels):
		super().__init__(xLabels, yLabels)
		self.Predictor = LinearRegression()
		return

	def _Predict(self, x:NDArray) -> NDArray:
		super()._Predict(x)

		predicted = self.Predictor.predict(x)

		return predicted

	def _Train(self, x:NDArray, y:NDArray) -> None:
		super()._Train(x, y)

		self.Predictor.fit(x, y)
		return


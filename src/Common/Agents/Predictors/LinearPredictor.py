from . import BasePredictor
from sklearn.linear_model import LinearRegression
from numpy.typing import NDArray
import typing
import src.Common.Enums.eDataColumnTypes as DCT
import src.Common.Utils.SharedCoreTypes as SCT

class LinearPredictor(BasePredictor.BasePredictor):

	def __init__(self,
			xLabels:typing.List[DCT.eDataColumnTypes],
			yLabels:typing.List[DCT.eDataColumnTypes],
			overrideConfig:SCT.Config):
		super().__init__(xLabels, yLabels, overrideConfig)

		self.Predictor = LinearRegression()
		return

	def _Predict(self, x:NDArray) -> NDArray:
		super()._Predict(x)

		predicted = self.Predictor.predict(x)

		return predicted

	def _Train(self, x:NDArray, y:NDArray) -> None:
		super()._Train(x, y)

		self.Predictor.fit(x, y)
		return True


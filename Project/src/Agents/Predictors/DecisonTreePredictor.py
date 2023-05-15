from . import BasePredictor
from xgboost import XGBClassifier
import numpy as np
import typing
import src.DataManager.DataColumnTypes as DCT
from numpy.typing import NDArray

class DecisonTreePredictor(BasePredictor.BasePredictor):

	def __init__(self,
			xLabels:typing.List[DCT.DataColumnTypes],
			yLabels:typing.List[DCT.DataColumnTypes]):
		super().__init__(xLabels, yLabels)
		self.Predictor = XGBClassifier(n_estimators=10, max_depth=5, learning_rate=1, objective='binary:logistic')

		return

	def _Predict(self, x:NDArray) -> NDArray:
		super()._Predict(x)

		predicted = self.Predictor.predict(x)

		return predicted

	def _Train(self, x:NDArray, y:NDArray) -> None:
		super()._Train(x, y)

		self.Predictor.fit(x, y)
		return


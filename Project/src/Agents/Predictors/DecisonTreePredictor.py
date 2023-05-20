from . import BasePredictor
import xgboost as xgb
import typing
import src.DataManager.DataColumnTypes as DCT
from numpy.typing import NDArray
import numpy as np

class DecisonTreePredictor(BasePredictor.BasePredictor):

	def __init__(self,
			xLabels:typing.List[DCT.DataColumnTypes],
			yLabels:typing.List[DCT.DataColumnTypes]):
		super().__init__(xLabels, yLabels)

		self.Predictor = None
		return

	def _Predict(self, x:NDArray) -> NDArray:
		super()._Predict(x)

		predicted = self.Predictor.predict(xgb.DMatrix(x))

		predicted = np.reshape(predicted, (len(predicted), -1))

		return predicted

	def _Train(self, x:NDArray, y:NDArray) -> None:
		super()._Train(x, y)

		dtrain = xgb.DMatrix(x, label=y)

		# regression problem
		params = {'objective': 'reg:squarederror'}

		# classification problem
		if self._IsDiscrete:
			params = {'objective': 'binary:logistic'}

		# Train the model
		num_rounds = 100
		self.Predictor = xgb.train(params, dtrain, num_rounds)

		return True


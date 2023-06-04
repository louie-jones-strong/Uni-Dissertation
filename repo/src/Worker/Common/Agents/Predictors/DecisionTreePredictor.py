from . import BasePredictor
import xgboost as xgb
import typing
import Common.DataManager.DataColumnTypes as DCT
from numpy.typing import NDArray
import numpy as np
import Common.Utils.SharedCoreTypes as SCT

class DecisionTreePredictor(BasePredictor.BasePredictor):

	def __init__(self,
			xLabels:typing.List[DCT.DataColumnTypes],
			yLabels:typing.List[DCT.DataColumnTypes],
			overrideConfig:SCT.Config):
		super().__init__(xLabels, yLabels, overrideConfig)

		self.Predictor = None
		return

	def _Predict(self, x:NDArray) -> NDArray:
		super()._Predict(x)

		dmatrix = xgb.DMatrix(x)
		predicted = self.Predictor.predict(dmatrix)

		predicted = np.reshape(predicted, (len(predicted), -1))

		return predicted

	def _Train(self, x:NDArray, y:NDArray) -> None:
		super()._Train(x, y)

		dtrain = xgb.DMatrix(x, label=y)

		params = {
			"max_depth": self.Config["max_depth"],
			"gamma": self.Config["gamma"]
		}


		# regression problem
		params["objective"] = "reg:squarederror"

		# classification problem
		if self._IsDiscrete:
			params["objective"] = "binary:logistic"

		# Train the model
		num_rounds = self.Config["num_rounds"]
		self.Predictor = xgb.train(params, dtrain, num_rounds)

		return True


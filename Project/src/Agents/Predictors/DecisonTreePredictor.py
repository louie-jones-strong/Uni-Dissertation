from . import BasePredictor
from xgboost import XGBClassifier
import numpy as np

class DecisonTreePredictor(BasePredictor.BasePredictor):

	def __init__(self, xLabel, yLabel):
		super().__init__(xLabel, yLabel)
		self.Predictor = XGBClassifier(n_estimators=10, max_depth=5, learning_rate=1, objective='binary:logistic')

		return

	def PredictValue(self, x):
		y = super().PredictValue(x)

		predictions = self.Predictor.predict(x)


		return predictions

	def Train(self, x, y):
		super().Train()

		self.Predictor.fit(x, y)

		return


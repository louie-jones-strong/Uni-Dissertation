from . import BasePredictor
from xgboost import XGBClassifier
import numpy as np

class DecisonTreePredictor(BasePredictor.BasePredictor):

	def __init__(self, xLabels, yLabels):
		super().__init__(xLabels, yLabels)
		self.Predictor = XGBClassifier(n_estimators=10, max_depth=5, learning_rate=1, objective='binary:logistic')

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


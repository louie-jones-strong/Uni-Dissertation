from . import BasePredictor


class EnsemblePredictor(BasePredictor.BasePredictor):

	def Predict(self, x):
		y, confidence, novelty = super().Predict(x)


		return y, confidence, novelty

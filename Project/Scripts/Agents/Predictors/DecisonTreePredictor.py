from . import BasePredictor


class EnsemblePredictor(BasePredictor.BasePredictor):

	def Predict(self, x):
		y, confidence, novelty = super().Predict(x)


		return y, confidence, novelty

	def Train(self):
		super().Train()

		x, y = self._GetSamples()

		return


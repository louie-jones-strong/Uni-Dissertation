from Agents.Predictors import BasePredictor
import numpy as np
from numpy.typing import NDArray
import typing
import DataManager.DataColumnTypes as DCT
import Utils.SharedCoreTypes as SCT


class MultiYPredictor(BasePredictor.BasePredictor):

	def __init__(self,
			xLabels:typing.List[DCT.DataColumnTypes],
			yLabels:typing.List[DCT.DataColumnTypes],
			overrideConfig:SCT.Config):
		super().__init__(xLabels, yLabels, overrideConfig)

		self._Predictors = {}
		for yLabel in yLabels:
			subConfig = self.Config["YPredictors"][yLabel.name]
			predictorName = subConfig["PredictorName"]

			self._Predictors[yLabel] = BasePredictor.GetPredictor(predictorName, xLabels, [yLabel], subConfig["PredictorConfig"])

		return

	def Observe(self, x, y):
		super().Observe(x, y)
		assert len(y) == len(self._YLabels), "y must have the same number of elements as yLabels"

		for i in range(len(self._YLabels)):
			yLabel = self._YLabels[i]
			self._Predictors[yLabel].Observe(x, [y[i]])

		return

	def _Evaluate(self, rawPrediction, proccessedY, y):

		data = rawPrediction[0]
		for i in range(1, len(rawPrediction)):
			data = np.concatenate((data, rawPrediction[i]), axis=1)

		error = abs(np.mean(proccessedY - data))


		totalItemWiseEqual = np.ones(len(rawPrediction[0]), dtype=np.bool_)
		for i in range(len(self._YLabels)):
			# accuracy
			label = self._YLabels[i]
			prediction = self._DataManager.PostProcessSingleColumn(rawPrediction[i], label)[0]

			itemWiseEqual = self._ItemWiseEqual(prediction, y[i])
			totalItemWiseEqual = np.logical_and(totalItemWiseEqual, itemWiseEqual)

		accuracy = np.mean(totalItemWiseEqual)

		return error, accuracy

	def _Predict(self, proccessedX:NDArray) -> NDArray:
		super()._Predict(proccessedX)

		predicted = []
		for yLabel in self._YLabels:
			prediction = self._Predictors[yLabel].CalRawPrediction(proccessedX)
			predicted.append(prediction)

		return predicted

	def _Train(self, x:NDArray, y:NDArray) -> None:
		super()._Train(x, y)

		wasTrained = False
		for yLabel in self._YLabels:

			if self._Predictors[yLabel].Train():
				wasTrained = True

		return wasTrained


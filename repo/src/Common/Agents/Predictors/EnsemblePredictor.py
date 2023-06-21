from . import BasePredictor
import numpy as np
from numpy.typing import NDArray
import typing
import src.Common.DataManager.DataColumnTypes as DCT
import src.Common.Utils.SharedCoreTypes as SCT

class EnsemblePredictor(BasePredictor.BasePredictor):

	def __init__(self,
			xLabels:typing.List[DCT.DataColumnTypes],
			yLabels:typing.List[DCT.DataColumnTypes],
			overrideConfig:SCT.Config):
		super().__init__(xLabels, yLabels, overrideConfig)


		subPredictorConfigs = self.Config["SubPredictors"]
		self._SubPredictors = []
		for config in subPredictorConfigs:
			predictorName = config["PredictorName"]
			subConfig = config["PredictorConfig"]

			predictor = BasePredictor.GetPredictor(predictorName, xLabels, yLabels, subConfig)
			self._SubPredictors.append(predictor)

		return


	def Observe(self, x, y) -> None:
		super().Observe(x, y)

		for predictor in self._SubPredictors:
			predictor.Observe(x, y)

		return


	def _Predict(self, proccessedX:NDArray) -> NDArray:
		super()._Predict(proccessedX)

		predictions = []

		for i in range(len(self._SubPredictors)):
			prediction = self._SubPredictors[i]._Predict(proccessedX)
			predictions.append(prediction)

		predictions = np.array(predictions)

		predictions = np.mean(predictions, axis=0)
		return predictions


	def _Train(self, x:NDArray, y:NDArray) -> bool:
		super()._Train(x, y)

		wasTrained = False
		for predictor in self._SubPredictors:

			if predictor.Train():
				wasTrained = True

		return wasTrained
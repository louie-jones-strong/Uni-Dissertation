import src.Common.Utils.ModelHelper as ModelHelper
from src.Common.Enums.eModelType import eModelType
import src.Common.Utils.Metrics.Logger as Logger


class Model:

	def __init__(self, modelType:eModelType):
		self._ModelType = modelType

		self._ModelHelper = ModelHelper.ModelHelper()
		modelData = self._ModelHelper.BuildModel(self._ModelType)
		self._Model, self._InputColumns, self._OutputColumns, self._ModelConfig = modelData

		self.HasTrainedModel = False

		self._Logger = Logger.Logger()

		self.UpdateModels()
		return

	def UpdateModels(self) -> None:
		self.HasTrainedModel = self._ModelHelper.FetchNewestWeights(self._ModelType, self._Model)
		print("fetched newest weights", self.HasTrainedModel)
		return

	def CanPredict(self) -> bool:
		return self.HasTrainedModel


	def _Predict(self, x):
		preProcessedX = self._ModelHelper.PreProcessColumns(x, self._InputColumns)

		y = self._Model.predict(preProcessedX, batch_size=len(x), verbose=0)

		postProcessedY = self._ModelHelper.PostProcessColumns(y, self._OutputColumns)

		return y, postProcessedY

	# abstract method
	# def Predict(self, in1, in2):
	# 	return out1, out2

	# abstract method
	# def FeedBack(self, x, y):
	# 	"""
	# 	calculates the models' loss and accuracy on the data collected from the environment
	# 	"""

	# 	y, postProcessedY = self._Predict(x)

	# 	y = self._ModelHelper.PreProcessColumns(y, self._OutputColumns)
	# 	return

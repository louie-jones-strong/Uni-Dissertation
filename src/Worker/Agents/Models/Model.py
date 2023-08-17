import src.Common.Utils.ModelHelper as ModelHelper
from src.Common.Enums.eModelType import eModelType
import src.Common.Utils.Metrics.Logger as Logger
import src.Common.Utils.ConfigHelper as ConfigHelper


class Model(ConfigHelper.ConfigurableClass):

	def __init__(self, modelType:eModelType):
		self.LoadConfig()
		self.ModelConfig = self.Config["ModelConfigs"][modelType.name]

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


	def Predict(self, x):
		preProcessedX = self._ModelHelper.PreProcessColumns(x, self._InputColumns)

		batchSize = min(len(x), self.ModelConfig["MaxDeploymentBatchSize"])
		rawY = self._Model.predict(preProcessedX, batch_size=batchSize, verbose=0)

		y = self._ModelHelper.PostProcessColumns(rawY, self._OutputColumns)

		return y, rawY

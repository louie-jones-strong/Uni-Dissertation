import typing
import src.Common.Utils.SharedCoreTypes as SCT
from src.Common.Enums.eModelType import eModelType
import numpy as np
from numpy.typing import NDArray
import src.Common.Utils.ModelHelper as ModelHelper

class ForwardModel:
	def __init__(self):

		self._ModelHelper = ModelHelper.ModelHelper()
		self._Model, self._InputColumns, self._OutputColumns, _ = self._ModelHelper.BuildModel(eModelType.Forward)
		self.HasTrainedModel = False

		self.UpdateModels()

		return

	def UpdateModels(self) -> None:
		self.HasTrainedModel = self._ModelHelper.FetchNewestWeights(eModelType.Forward, self._Model)
		print("fetched newest weights", self.HasTrainedModel)

		return

	def CanPredict(self) -> bool:
		return self.HasTrainedModel

	def Predict(self,
			states:SCT.State_List,
			actions:SCT.Action_List
			) -> typing.Tuple[SCT.State_List, SCT.Reward_List, NDArray[np.bool_]]:

		x = self._ModelHelper.PreProcessColumns([states, actions], self._InputColumns)

		y = self._Model.predict(x, batch_size=len(states), verbose=0)

		nextStates = self._ModelHelper.PostProcessSingleColumn(y[0], self._OutputColumns[0])[0]
		rewards = self._ModelHelper.PostProcessSingleColumn(y[1], self._OutputColumns[1])[0]
		terminateds = self._ModelHelper.PostProcessSingleColumn(y[2], self._OutputColumns[2])[0]



		return nextStates, rewards, terminateds
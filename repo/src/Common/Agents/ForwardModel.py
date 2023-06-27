import typing
from typing import Optional
import src.Common.Utils.SharedCoreTypes as SCT
import src.Common.Enums.DataColumnTypes as DCT
from src.Common.Enums.ModelType import ModelType
import numpy as np
from numpy.typing import NDArray
import src.Common.Utils.ModelHelper as ModelHelper

class ForwardModel:
	def __init__(self, envSim:Optional[object]):

		self._EnvSimulation = envSim

		self._ModelHelper = ModelHelper.ModelHelper()
		self._Model, self._InputColumns, self._OutputColumns = self._ModelHelper.BuildModel(ModelType.Forward)

		print("fetching newest weights")
		didFetch = self._ModelHelper.FetchNewestWeights(ModelType.Forward, self._Model)
		print("fetched newest weights", didFetch)



		self._SimulatedStates = 0
		return

	def CanPredict(self) -> bool:
		if self._EnvSimulation is not None:
			return True

		return True # todo if no model loaded return false


	def Predict(self,
			states:SCT.State_List,
			actions:SCT.Action_List
			) -> typing.Tuple[SCT.State_List, SCT.Reward_List, NDArray[np.bool_], NDArray[np.bool_]]:

		if self._EnvSimulation is not None:
			nextStates, rewards, terminateds = self._EnvSimulation.Predict(states, actions)

		else:
			x = self._ModelHelper.PreProcessColumns([states, actions], self._InputColumns)

			y = self._Model.predict(x, batch_size=len(states), verbose=0)

			nextStates = self._ModelHelper.PostProcessSingleColumn(y[0], self._OutputColumns[0])[0]
			rewards = self._ModelHelper.PostProcessSingleColumn(y[1], self._OutputColumns[1])[0]
			terminateds = self._ModelHelper.PostProcessSingleColumn(y[2], self._OutputColumns[2])[0]



		return nextStates, rewards, terminateds
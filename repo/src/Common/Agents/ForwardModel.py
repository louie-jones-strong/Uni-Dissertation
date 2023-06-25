import typing
from typing import Optional
import src.Common.Utils.SharedCoreTypes as SCT
import src.Common.Enums.DataColumnTypes as DCT
import numpy as np
from numpy.typing import NDArray

class ForwardModel:
	def __init__(self, envSim:Optional[object],
			overrideConfig:SCT.Config):
		self._EnvSimulation = envSim

		xColumns = [
			DCT.DataColumnTypes.CurrentState,
			DCT.DataColumnTypes.Action]

		yColumns = [
			DCT.DataColumnTypes.NextState,
			DCT.DataColumnTypes.Reward,
			DCT.DataColumnTypes.Terminated,
			DCT.DataColumnTypes.Truncated]

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
			nextStates, rewards, terminateds, truncateds = self._EnvSimulation.Predict(states, actions)

		else:
			values, confidences = self._PredictiveModel.Predict([states, actions])
			nextStates, rewards, terminateds, truncateds = values

			nextStates = nextStates[0]
			rewards = rewards[0]
			terminateds = terminateds[0]
			truncateds = truncateds[0]

		return nextStates, rewards, terminateds, truncateds
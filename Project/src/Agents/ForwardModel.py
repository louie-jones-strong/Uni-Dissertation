import typing
import src.Utils.SharedCoreTypes as SCT
import src.DataManager.DataColumnTypes as DCT
import numpy as np
from numpy.typing import NDArray
import src.Agents.Predictors.MultiYPredictor as MultiYPredictor

class ForwardModel:
	def __init__(self, envSim):
		self._EnvSimulation = envSim

		xColumns = [
			DCT.DataColumnTypes.CurrentState,
			DCT.DataColumnTypes.Action]

		yColumns = [
			DCT.DataColumnTypes.NextState,
			DCT.DataColumnTypes.Reward,
			DCT.DataColumnTypes.Terminated,
			DCT.DataColumnTypes.Truncated]

		if self._EnvSimulation is None:
			self._PredictiveModel = MultiYPredictor.MultiYPredictor(xColumns, yColumns)

		self._SimulatedStates = 0
		return

	def Remember(self,
			state:SCT.State,
			action:SCT.Action,
			reward:SCT.Reward,
			nextState:SCT.State,
			terminated:bool,
			truncated:bool) -> None:

		if self._EnvSimulation is None:
			self._PredictiveModel.Observe([[state], [action]], [[nextState], [reward], [terminated], [truncated]])
		return


	def Predict(self,
			states:SCT.State_List,
			actions:SCT.Action_List
			) -> typing.Tuple[SCT.State_List, SCT.Reward_List, NDArray[np.bool_], NDArray[np.bool_]]:

		if self._EnvSimulation is not None:
			nextStates, rewards, terminateds, truncateds = self._EnvSimulation.Predict(states, actions)

		else:
			nextStates, rewards, terminateds, truncateds = self._PredictiveModel.Predict([states, actions])


		return nextStates, rewards, terminateds, truncateds
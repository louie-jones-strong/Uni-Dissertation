import typing
import src.Common.Utils.SharedCoreTypes as SCT
from src.Common.Enums.eModelType import eModelType
import numpy as np
from numpy.typing import NDArray
import src.Worker.Agents.Models.Model as Model

class ForwardModel(Model.Model):
	def __init__(self):
		super().__init__(eModelType.Forward)
		return

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

	def Remember(self,
			state:SCT.State,
			action:SCT.Action,
			reward:SCT.Reward,
			nextState:SCT.State,
			terminated:bool,
			truncated:bool) -> None:
		"""
		calculates the models' loss and accuracy on the data collected from the environment
		"""

		x = self._ModelHelper.PreProcessColumns([state, action], self._InputColumns)
		target = self._ModelHelper.PreProcessColumns([nextState, reward, terminated], self._OutputColumns)
		postTarget = target

		predictions = self._Model(x)

		metrics = self._ModelHelper.CalculateModelMetrics(self._OutputColumns, predictions, target, postTarget)
		_, losses, accuracies = metrics


		# log the metrics
		logDict = {}
		for i in range(len(self._OutputColumns)):
			col = self._OutputColumns[i]
			loss = losses[i]
			accuracy = accuracies[i]

			logDict[f"Val_{col.name}_Loss"] = loss

			if accuracy is not None:
				logDict[f"Val_{col.name}_Accuracy"] = accuracy

		self._Logger.LogDict(logDict)

		return

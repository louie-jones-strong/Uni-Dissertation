import typing
import src.Common.Utils.SharedCoreTypes as SCT
from src.Common.Enums.eModelType import eModelType
import numpy as np
from numpy.typing import NDArray
import src.Worker.Agents.Models.Model as Model

class ForwardModel(Model.Model):
	def __init__(self):
		self.StateModel = Model.Model(eModelType.Forward_NextState)
		self.RewardModel = Model.Model(eModelType.Forward_Reward)
		self.TerminatedModel = Model.Model(eModelType.Forward_Terminated)
		return

	def UpdateModels(self) -> None:
		self.StateModel.UpdateModels()
		self.RewardModel.UpdateModels()
		self.TerminatedModel.UpdateModels()
		return

	def CanPredict(self) -> bool:
		return self.StateModel.CanPredict() and self.RewardModel.CanPredict() and self.TerminatedModel.CanPredict()

	def Predict(self,
			states:SCT.State_List,
			actions:SCT.Action_List
			) -> typing.Tuple[SCT.State_List, SCT.Reward_List, NDArray[np.bool_]]:

		x = [states, actions]
		nextStates, _ = self.StateModel.Predict(x)
		rewards, _ = self.RewardModel.Predict(x)
		terminateds, _ = self.TerminatedModel.Predict(x)

		return nextStates[0], rewards[0], terminateds[0]

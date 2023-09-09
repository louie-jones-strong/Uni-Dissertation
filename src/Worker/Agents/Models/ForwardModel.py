import typing
import src.Common.Utils.SharedCoreTypes as SCT
from src.Common.Enums.eModelType import eModelType
import numpy as np
from numpy.typing import NDArray
import src.Worker.Agents.Models.Model as Model
import src.Worker.Environments.BaseEnv as BaseEnv

class ForwardModel(Model.Model):
	def __init__(self) -> None:
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
		if self.StateModel.Config["UseRealSim"]:
			return True

		return self.StateModel.CanPredict() and self.RewardModel.CanPredict() and self.TerminatedModel.CanPredict()

	def Predict(self,
			states:SCT.State_List,
			envs:typing.List[BaseEnv.BaseEnv],
			actions:SCT.Action_List
			) -> typing.Tuple[SCT.State_List, typing.List[BaseEnv.BaseEnv], SCT.Reward_List, NDArray[np.bool_]]:

		if self.StateModel.Config["UseRealSim"]:
			# simulate step using real envs

			nextStates = []
			nextEnvs = []
			rewards = []
			terminateds = []

			for i in range(len(envs)):
				# check if the env is done
				if envs[i].IsDone():
					nextEnvs.append(envs[i])
					nextStates.append(states[i])
					rewards.append(0.0)
					terminateds.append(True)
					continue

				nextEnv = envs[i].Clone()
				transition = nextEnv.Step(actions[i])

				nextEnvs.append(nextEnv)
				nextStates.append(transition[0])
				rewards.append(transition[1])
				terminateds.append(transition[2])


		else:  # predict step using models
			nextStates, rewards, terminateds = self.PredictStep(states, actions)
			nextEnvs = [None] * len(envs)



		return nextStates, nextEnvs, rewards, terminateds

	def PredictStep(self,
			states:SCT.State_List,
			actions:SCT.Action_List) -> typing.Tuple[SCT.State_List, SCT.Reward_List, NDArray[np.bool_]]:

		x = [states, actions]
		nextStates, _ = self.StateModel.Predict(x)
		rewards, _ = self.RewardModel.Predict(x)
		terminateds, _ = self.TerminatedModel.Predict(x)

		return nextStates[0], rewards[0], terminateds[0]
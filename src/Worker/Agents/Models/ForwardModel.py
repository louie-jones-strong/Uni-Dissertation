import typing
import src.Common.Utils.SharedCoreTypes as SCT
from src.Common.Enums.eModelType import eModelType
import numpy as np
from numpy.typing import NDArray
import src.Worker.Agents.Models.Model as Model
import src.Worker.Environments.BaseEnv as BaseEnv
import src.Common.Utils.Metrics.Metrics as Metrics

class ForwardModel(Model.Model):
	def __init__(self) -> None:
		self.ForwardModel = Model.Model(eModelType.Forward)
		self.StateModel = Model.Model(eModelType.Forward_NextState)
		self.RewardModel = Model.Model(eModelType.Forward_Reward)
		self.TerminatedModel = Model.Model(eModelType.Forward_Terminated)
		self._Metrics = Metrics.Metrics()
		return

	def UpdateModels(self) -> None:
		self.ForwardModel.UpdateModels()
		self.StateModel.UpdateModels()
		self.RewardModel.UpdateModels()
		self.TerminatedModel.UpdateModels()
		return

	def CanPredict(self) -> bool:
		if self.StateModel.Config["UseRealSim"]:
			return True

		combinedModel = self.ForwardModel.CanPredict()
		splitModel = self.StateModel.CanPredict() and \
			self.RewardModel.CanPredict() and \
			self.TerminatedModel.CanPredict()

		return combinedModel or splitModel

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


				with self._Metrics.Time("EnvClone"):
					nextEnv = envs[i].Clone()

				with self._Metrics.Time("EnvStep"):
					transition = nextEnv.Step(actions[i])

				nextEnvs.append(nextEnv)
				nextStates.append(transition[0])
				rewards.append(transition[1])
				terminateds.append(transition[2])


		else:  # predict step using models

			if self.ForwardModel.CanPredict():
				x = [states, actions]
				y, _ = self.ForwardModel.Predict(x)
				nextStates = y[0][0]
				rewards = y[1][0]
				terminateds = y[2][0]

				nextEnvs = [None] * len(envs)


			else:
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
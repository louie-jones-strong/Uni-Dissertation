import src.Worker.Agents.BaseAgent as BaseAgent
import src.Common.Utils.SharedCoreTypes as SCT
import src.Worker.Environments.BaseEnv as BaseEnv
import typing
import numpy as np


class HardCodedAi(BaseAgent.BaseAgent):

	def __init__(self, envConfig:SCT.Config, isTrainingMode:bool):
		super().__init__(envConfig, isTrainingMode)

		self.BehaviorsLookups = {
			"Normal": [
				2, 1, 1, 0, 2, 1, 0,
				2, 1, 1, 0, 0, 1, 0,
				2, 1, 1, 0, 0, 1, 0,
				2, 2, 1, 0, 0, 0, 0
			],
			"Human": [
				2, 1, 1, 0, 2, 1, 0,
				2, 1, 1, 0, 0, 1, 0,
				2, 1, 1, 0, 0, 1, 0,
				2, 2, 1, 0, 0, 0, 0
			],
			"PlayStyle": [
				1, 0, 0, 0, 2, 1, 0,
				2, 1, 1, 0, 0, 1, 0,
				2, 1, 1, 0, 0, 1, 0,
				2, 2, 1, 0, 0, 0, 0
			],
		}
		return

	def Reset(self) -> None:
		super().Reset()
		return

	def GetAction(self, state:SCT.State, env:BaseEnv.BaseEnv) -> typing.Tuple[SCT.Action, str]:
		super().GetAction(state, env)

		playStyleWeights = self.Config["HardcodedConfig"]["PlayStyleWeights"]

		actions = []
		actionWeights = []

		for key, value in self.BehaviorsLookups.items():

			actions.append(value[state])
			actionWeights.append(playStyleWeights[key])


		# normalize weights
		actionWeights = np.array(actionWeights)
		actionWeights = actionWeights / np.sum(actionWeights)


		action = np.random.choice(actions, p=actionWeights)


		reason = {
			"Actions": actions,
			"ActionWeights": actionWeights,
			"AgentType": "HardCoded"
		}

		return action, reason
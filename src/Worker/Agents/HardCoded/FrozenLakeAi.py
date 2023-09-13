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

	def GetAction(self, state:SCT.State, env:BaseEnv.BaseEnv) -> \
			typing.Tuple[SCT.Action, SCT.ActionValues, SCT.ActionReason]:
		super().GetAction(state, env)

		playStyleWeights = self.Config["HardcodedConfig"]["PlayStyleWeights"]

		action = 0
		maxWeight = 0

		for key, value in self.BehaviorsLookups.items():

			if playStyleWeights[key] > maxWeight:
				maxWeight = playStyleWeights[key]
				action = value[state]


		reason = {
			"AgentType": "HardCoded"
		}

		actionValues = np.zeros(self.ActionSpace.n).astype(np.float32)
		return action, actionValues, reason
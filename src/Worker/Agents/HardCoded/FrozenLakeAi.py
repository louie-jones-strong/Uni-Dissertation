import src.Worker.Agents.BaseAgent as BaseAgent
import src.Common.Utils.SharedCoreTypes as SCT
import src.Worker.Environments.BaseEnv as BaseEnv
import typing



class HardCodedAi(BaseAgent.BaseAgent):

	def __init__(self, envConfig:SCT.Config, isTrainingMode:bool):
		super().__init__(envConfig, isTrainingMode)

		self.ActionLookup = [
			2, 1, 1, 0, 2, 1, 0,
			2, 1, 1, 0, 0, 1, 0,
			2, 1, 1, 0, 0, 1, 0,
			2, 2, 1, 0, 0, 0, 0
		]
		return

	def Reset(self) -> None:
		super().Reset()
		return

	def GetAction(self, state:SCT.State, env:BaseEnv.BaseEnv) -> typing.Tuple[SCT.Action, str]:
		super().GetAction(state, env)

		action = self.ActionLookup[state]

		return action, "HardCoded"
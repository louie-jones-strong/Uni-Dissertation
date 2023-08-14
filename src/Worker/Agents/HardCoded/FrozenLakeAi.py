import src.Worker.Agents.BaseAgent as BaseAgent
import src.Common.Utils.SharedCoreTypes as SCT
import typing



class HardCodedAi(BaseAgent.BaseAgent):

	def __init__(self, envConfig:SCT.Config, isTrainingMode:bool):
		super().__init__(envConfig, isTrainingMode)
		return

	def Reset(self) -> None:
		super().Reset()
		return

	def Save(self, path:str) -> None:
		super().Save(path)
		return

	def Load(self, path:str) -> None:
		super().Load(path)
		return



	def Remember(self,
		state:SCT.State,
		action:SCT.Action,
		reward:SCT.Reward,
		nextState:SCT.State,
		terminated:bool,
		truncated:bool) -> None:

		super().Remember(state, action, reward, nextState, terminated, truncated)
		return


	def GetAction(self, state:SCT.State) -> typing.Tuple[SCT.Action, str]:
		super().GetAction(state)

		return 0, "HardCoded"
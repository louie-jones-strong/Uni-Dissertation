import src.Worker.Agents.BaseAgent as BaseAgent
import src.Common.Utils.SharedCoreTypes as SCT
import src.Worker.Environments.BaseEnv as BaseEnv
import typing


class HardCodedAi(BaseAgent.BaseAgent):

	def __init__(self, envConfig:SCT.Config, isTrainingMode:bool):
		super().__init__(envConfig, isTrainingMode)

		# config
		self.PosWeighting = 0.2
		self.AngleWeighting = 1
		self.PredictionDeltaTime = 1
		return

	def Reset(self) -> None:
		super().Reset()
		return

	def GetAction(self, state:SCT.State, env:BaseEnv.BaseEnv) -> typing.Tuple[SCT.Action, str]:
		super().GetAction(state, env)

		if state is None:
			return 0, "None State"

		cartPos = state[0]
		cartVel = state[1]
		poleAngle = state[2]
		poleVel = state[3]

		predPos = cartPos + (cartVel * self.PredictionDeltaTime)
		predAngle = poleAngle + (poleVel * self.PredictionDeltaTime)


		actionWeight = 0


		# center the cart
		posWeight = predPos
		posWeight *= self.PosWeighting
		actionWeight += posWeight

		# keep the pole upright
		angleWeight = predAngle
		angleWeight *= self.AngleWeighting
		actionWeight += angleWeight

		# choice Action
		action = 0
		if actionWeight > 0:
			action = 1

		return action, "HardCoded"
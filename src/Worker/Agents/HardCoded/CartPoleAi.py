import src.Worker.Agents.BaseAgent as BaseAgent
import src.Common.Utils.SharedCoreTypes as SCT
import src.Worker.Environments.BaseEnv as BaseEnv
import typing


class HardCodedAi(BaseAgent.BaseAgent):

	def __init__(self, envConfig:SCT.Config, isTrainingMode:bool):
		super().__init__(envConfig, isTrainingMode)

		# config
		self.PosWeighting = 0.1
		self.AngleWeighting = 4
		self.PredictionDeltaTime = 0.5

		self.BehaviorsOffsets = {
			"Normal": 0,
			"Human": 0,
			"PlayStyle": 1,
		}
		return

	def Reset(self) -> None:
		super().Reset()
		return




	def GetAction(self, state:SCT.State, env:BaseEnv.BaseEnv) -> typing.Tuple[SCT.Action, str]:
		super().GetAction(state, env)

		if state is None:
			return 0, "None State"

		offset = 0
		playStyleWeights = self.Config["HardcodedConfig"]["PlayStyleWeights"]
		for key, value in self.BehaviorsOffsets.items():
			offset += playStyleWeights[key] * value

		directionWeights = self.CalMoveWeighting(state, offset)


		# choice Action
		action = 0
		if directionWeights > 0:
			action = 1

		return action, "HardCoded"


	def CalMoveWeighting(self, state:SCT.State, offset:float):
		cartPos = state[0]
		cartVel = state[1]
		poleAngle = state[2]
		poleVel = state[3]

		predPos = cartPos + (cartVel * self.PredictionDeltaTime)
		predAngle = poleAngle + (poleVel * self.PredictionDeltaTime)




		actionWeight = 0


		# center the cart
		posWeight = predPos + offset
		posWeight *= self.PosWeighting
		actionWeight += posWeight





		# keep the pole upright
		angleWeight = predAngle
		angleWeight *= self.AngleWeighting
		actionWeight += angleWeight

		return actionWeight
import src.Common.Utils.SharedCoreTypes as SCT
from src.Common.Enums.eModelType import eModelType
import src.Worker.Agents.Models.ValueModel as ValueModel
import src.Worker.Agents.Models.PlayStyleModel as PlayStyleModel


class CombinedValueModel:
	def __init__(self):
		self.RewardModel = ValueModel.ValueModel(eModelType.Value)
		self.PlayStyleModel = PlayStyleModel.PlayStyleModel(eModelType.PlayStyleDiscriminator)
		return

	def CanPredict(self) -> bool:
		return self.RewardModel.CanPredict() and self.PlayStyleModel.CanPredict()


	def Predict(self, states:SCT.State_List, actions:SCT.Action_List):

		if not self.CanPredict():
			return

		rewardValues = self.RewardModel.Predict(states)
		playStyleValues = self.PlayStyleModel.Predict(states, actions)

		values = rewardValues + playStyleValues
		values /= 2

		return values
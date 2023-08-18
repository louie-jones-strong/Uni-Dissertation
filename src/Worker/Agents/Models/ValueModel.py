import typing
import src.Common.Utils.SharedCoreTypes as SCT
from src.Common.Enums.eModelType import eModelType
import src.Worker.Agents.Models.Model as Model

class ValueModel(Model.Model):
	def __init__(self):
		super().__init__(eModelType.Value)
		return

	def Predict(self, states:SCT.State_List) -> typing.Tuple[SCT.Reward_List]:

		x = [states]
		values, _ = super().Predict(x)

		return values[0]
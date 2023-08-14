import typing
import src.Common.Utils.SharedCoreTypes as SCT
from src.Common.Enums.eModelType import eModelType
import numpy as np
from numpy.typing import NDArray
import src.Worker.Agents.Models.Model as Model

class ValueModel(Model.Model):
	def __init__(self):
		super().__init__(eModelType.Value)
		return

	def Predict(self, states:SCT.State_List) -> typing.Tuple[SCT.State_List, SCT.Reward_List, NDArray[np.bool_]]:

		x = self._ModelHelper.PreProcessSingleColumn(states, self._InputColumns[0])

		y = self._Model.predict(x, batch_size=len(states), verbose=0)

		values = self._ModelHelper.PostProcessSingleColumn(y[0], self._OutputColumns[0])[0]

		return values
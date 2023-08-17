import typing
import src.Common.Utils.SharedCoreTypes as SCT
from src.Common.Enums.eModelType import eModelType
import numpy as np
from numpy.typing import NDArray
import src.Worker.Agents.Models.Model as Model

class PlayStyleModel(Model.Model):
	def __init__(self):
		super().__init__(eModelType.PlayStyleDiscriminator)
		return

	def Predict(self,
			states:SCT.State_List,
			actions:SCT.Action_List) -> typing.Tuple[SCT.State_List, SCT.Reward_List, NDArray[np.bool_]]:

		x = self._ModelHelper.PreProcessColumns([states, actions], self._InputColumns)

		y = self._Model.predict(x, batch_size=len(states), verbose=0)

		values = self._ModelHelper.PostProcessSingleColumn(y[0], self._OutputColumns[0])[0]

		return values
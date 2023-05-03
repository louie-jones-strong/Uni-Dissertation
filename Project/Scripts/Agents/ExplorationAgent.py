#region typing dependencies
from typing import TYPE_CHECKING, Any, Optional, Type, TypeVar

import Utils.SharedCoreTypes as SCT

from numpy.typing import NDArray
if TYPE_CHECKING:
	pass
# endregion

# other file dependencies

from . import BaseAgent
import numpy as np

class ExplorationAgent(BaseAgent.BaseAgent):


	def GetAction(self, state:SCT.State) -> SCT.Action:
		super().GetAction(state)
		actionValues = self.GetActionValues(state)
		return self._GetMaxValues(actionValues)

	def GetActionValues(self, state:SCT.State) -> NDArray[np.float32]:
		super().GetActionValues(state)


		novelties, values = self.DataManager._MarkovModel.GetStateInfo(state)

		if self.Mode == BaseAgent.AgentMode.Train:
			return novelties


		return values
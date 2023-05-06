import numpy as np
import src.Utils.SharedCoreTypes as SCT
from numpy.typing import NDArray

from . import BaseAgent

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
from . import BaseAgent
import numpy as np

class ExplorationAgent(BaseAgent.BaseAgent):


	def GetAction(self, state):
		super().GetAction(state)
		actionValues = self.GetActionValues(state)
		return self._GetMaxValues(actionValues)

	def GetActionValues(self, state):
		super().GetActionValues(state)


		novelties, values = self.DataManager._MDM.GetStateInfo(state)

		if self.Mode == BaseAgent.AgentMode.Train:
			return novelties


		return values
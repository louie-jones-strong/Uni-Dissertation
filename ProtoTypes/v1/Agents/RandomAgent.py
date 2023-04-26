from . import BaseAgent
import numpy as np

class RandomAgent(BaseAgent.BaseAgent):

	def GetActionValues(self, state):
		actionValues = np.random.rand(self.Env.ActionSpace.n)
		return actionValues
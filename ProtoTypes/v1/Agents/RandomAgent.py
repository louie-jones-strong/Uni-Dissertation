from . import BaseAgent
import numpy as np

class RandomAgent(BaseAgent.BaseAgent):

	def GetActionValues(self, state):
		actionValues = np.random.rand(self.Env.action_space.n)
		return actionValues
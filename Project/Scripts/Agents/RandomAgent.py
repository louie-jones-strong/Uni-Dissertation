from . import BaseAgent
import numpy as np

class RandomAgent(BaseAgent.BaseAgent):


	def GetAction(self, state):
		super().GetAction(state)
		return self.Env.ActionSpace.sample()


	def GetActionValues(self, state):
		super().GetActionValues(state)
		return np.random.rand(self.Env.ActionSpace.n)
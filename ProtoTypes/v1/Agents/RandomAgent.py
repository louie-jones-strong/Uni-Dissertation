from . import BaseAgent
import numpy as np

class RandomAgent(BaseAgent.BaseAgent):

	def GetActionValues(self, state):

		# minReward = self.Env.RewardRange[0]
		# maxReward = self.Env.RewardRange[1]

		# actionValues = np.random.rand(self.Env.ActionSpace.n)
		# actionValues *= np.ones(self.Env.ActionSpace.n) * (maxReward - minReward)
		# actionValues += np.ones(self.Env.ActionSpace.n) * minReward

		return np.random.rand(self.Env.ActionSpace.n)
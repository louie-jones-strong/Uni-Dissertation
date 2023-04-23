from . import BaseAgent

class RandomAgent(BaseAgent.BaseAgent):
	def __init__(self, env):
		super().__init__(env)
		return

	def GetAction(self, state):
		return self.Env.action_space.sample()
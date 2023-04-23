from . import BaseAgent
import Utils.Network as Network
import numpy as np

class DQNAgent(BaseAgent.BaseAgent):
	def __init__(self, env):
		super().__init__(env)
		self.Network = Network.Network()

		inputShape = self.Env.observation_space.shape
		self.Network.BuildModel(inputShape, self.Env.action_space.n)


		return

	def GetActionValues(self, state):
		state = np.expand_dims(state, axis=0)
		actionValues = self.Network.Predict(state)

		return actionValues
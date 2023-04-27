from . import BaseAgent
import numpy as np


class MonteCarloAgent(BaseAgent.BaseAgent):

	def __init__(self, env, envConfig, mode=BaseAgent.AgentMode.Train):
		super().__init__(env, envConfig, mode=mode)

		self._SubAgent = BaseAgent.GetAgent(self.Config["SubAgent"])(self.Env, envConfig)#, mode=BaseAgent.AgentMode.Play)
		self._EnvSim = None
		return

	def GetActionValues(self, state):


		actionValues = self._SearchActions(self.Env, state)



		return actionValues

	def _SearchActions(self, env, state, currentDepth=0):

		actionValues = self._SubAgent.GetActionValues(state)

		if currentDepth >= self.Config["MaxDepth"]:
			return actionValues

		actionPrioList = np.argsort(actionValues)[::-1]

		for i in range(self.Config["TopActionCount"]):
			action = actionPrioList[i]

			envCopy = env.Clone()
			nextState, reward, done = envCopy.Step(action)

			if done:
				actionValues[action] = reward

			newActionValues = self._SearchActions(envCopy, nextState, currentDepth=currentDepth + 1)

			actionValues[action] += np.max(newActionValues)

			del envCopy


		return actionValues





	def Reset(self):
		super().Reset()
		self._SubAgent.Reset()
		return

	def Remember(self, state, action, reward, nextState, done):
		super().Remember(state, action, reward, nextState, done)
		self._SubAgent.Remember(state, action, reward, nextState, done)
		return

	def Save(self, path):
		super().Save(path)
		self._SubAgent.Save(path)
		self._EnvSim.Save(path)
		return

	def Load(self, path):
		super().Load(path)
		self._SubAgent.Load(path)
		self._EnvSim.Load(path)
		return

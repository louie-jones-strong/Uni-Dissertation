from . import BaseAgent
import numpy as np


class MonteCarloAgent(BaseAgent.BaseAgent):

	def __init__(self, env, envConfig, mode=BaseAgent.AgentMode.Train):
		super().__init__(env, envConfig, mode=mode)

		self._SubAgent = BaseAgent.GetAgent(self.Config["SubAgent"])(self.Env, envConfig, mode=mode)

		return

	def GetAction(self, state):
		super().GetAction(state)
		actionValues = self.GetActionValues(state)
		return self._GetMaxValues(actionValues)

	def GetActionValues(self, state):
		super().GetActionValues(state)

		return self._SearchActions(self.Env, state)

	def _SearchActions(self, env, state, depth=0):

		actionValues = self._SubAgent.GetActionValues(state)

		if depth >= self.Config["MaxDepth"]:
			return actionValues * self.Config["DiscountFactor"]

		actionPrioList = np.argsort(actionValues)[::-1]

		for i in range(self.Config["TopActionCount"]):
			action = actionPrioList[i]

			# predict with markov model
			nextState, reward, terminated = self.DataManager._MarkovModel.Predict(state, action)

			# if not in markov model, simulate
			if terminated is None:

				envCopy = env.Clone()
				nextState, reward, terminated, truncated = envCopy.Step(action)

				actionValues[action] = reward

				if not (terminated or truncated):
					actionValues[action] += np.max(self._SearchActions(envCopy, nextState, depth=depth + 1))

				del envCopy

			else:
				actionValues[action] = reward

				if not terminated:
					actionValues[action] += np.max(self._SearchActions(env, nextState, depth=depth + 1))


		return actionValues * self.Config["DiscountFactor"]




	def Reset(self):
		super().Reset()
		self._SubAgent.Reset()
		return

	def Remember(self, state, action, reward, nextState, terminated, truncated):
		super().Remember(state, action, reward, nextState, terminated, truncated)
		self._SubAgent.Remember(state, action, reward, nextState, terminated, truncated)
		return

	def Save(self, path):
		super().Save(path)
		self._SubAgent.Save(path)
		return

	def Load(self, path):
		super().Load(path)
		self._SubAgent.Load(path)
		return

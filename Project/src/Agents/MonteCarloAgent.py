import src.Agents.BaseAgent as BaseAgent
import numpy as np
import src.Utils.SharedCoreTypes as SCT
from src.Environments.BaseEnv import BaseEnv
from numpy.typing import NDArray


class MonteCarloAgent(BaseAgent.BaseAgent):

	def __init__(self,
			observationSpace:SCT.StateSpace,
			actionSpace:SCT.ActionSpace,
			envConfig:SCT.Config,
			mode:BaseAgent.AgentMode = BaseAgent.AgentMode.Train):
		super().__init__(observationSpace, actionSpace, envConfig, mode=mode)

		self._SubAgent = BaseAgent.GetAgent(self.Config["SubAgent"])(self._ObservationSpace, self._ActionSpace, mode=mode)

		return

	def GetAction(self, state:SCT.State) -> SCT.Action:
		super().GetAction(state)
		actionValues = self.GetActionValues(state)
		return self._GetMaxValues(actionValues)

	def GetActionValues(self, state:SCT.State) -> NDArray[np.float32]:
		super().GetActionValues(state)

		return self._SearchActions(self.Env, state)

	def _SearchActions(self, env:BaseEnv, state:SCT.State, depth:int = 0) -> NDArray[np.float32]:

		actionValues = self._SubAgent.GetActionValues(state)
		dicountFactor:float = self.Config["DiscountFactor"]

		if depth >= self.Config["MaxDepth"]:
			return actionValues * dicountFactor

		actionPrioList = np.argsort(actionValues)[::-1]

		for i in range(self.Config["TopActionCount"]):
			action = actionPrioList[i]

			# predict with markov model
			prediction = self.DataManager._MarkovModel.Predict(state, action)


			# if not in markov model, simulate
			if prediction is None:

				envCopy = env.Clone()
				nextState, reward, terminated, truncated = envCopy.Step(action)

				actionValues[action] = reward

				if not (terminated or truncated):
					actionValues[action] += np.max(self._SearchActions(envCopy, nextState, depth=depth + 1))

			else:
				nextState, reward, terminated, truncated = prediction

				actionValues[action] = reward

				if not terminated:
					actionValues[action] += np.max(self._SearchActions(env, nextState, depth=depth + 1))


		return actionValues * dicountFactor

	def Reset(self) -> None:
		super().Reset()
		self._SubAgent.Reset()
		return

	def Remember(self,
			state:SCT.State,
			action:SCT.Action,
			reward:SCT.Reward,
			nextState:SCT.State,
			terminated:bool,
			truncated:bool) -> None:

		super().Remember(state, action, reward, nextState, terminated, truncated)
		self._SubAgent.Remember(state, action, reward, nextState, terminated, truncated)
		return

	def Save(self, path:str) -> None:
		super().Save(path)
		self._SubAgent.Save(path)
		return

	def Load(self, path:str) -> None:
		super().Load(path)
		self._SubAgent.Load(path)
		return

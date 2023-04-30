from . import BaseAgent
from DataManger.MarkovDecisionProcess import MarkovDecisionProcess as MDP
import numpy as np

class ExplorationAgent(BaseAgent.BaseAgent):

	def __init__(self, env, config, mode=BaseAgent.AgentMode.Train):
		super().__init__(env, config, mode=mode)

		self.Mdp = MDP(self.Env.ActionSpace.n)
		return

	def GetAction(self, state):
		super().GetAction(state)
		actionValues = self.GetActionValues(state)
		return self._GetMaxValues(actionValues)

	def GetActionValues(self, state):
		super().GetActionValues(state)

		novelties, values = self.Mdp.GetStateInfo(state)

		if self.Mode == BaseAgent.AgentMode.Train:
			return novelties


		return novelties


	def Remember(self, state, action, reward, nextState, terminated, truncated):
		super().Remember(state, action, reward, nextState, terminated, truncated)
		self.Mdp.Remember(state, action, reward, nextState, terminated, truncated)
		return

	def Reset(self):
		super().Reset()
		self.Mdp.Reset()
		return

	def Save(self, path):
		super().Save(path)
		self.Mdp.Save(path)
		return

	def Load(self, path):
		super().Load(path)
		self.Mdp.Load(path)
		return
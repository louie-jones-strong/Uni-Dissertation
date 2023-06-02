import os
import time
import typing
from collections import deque

import Agents.BaseAgent as BaseAgent
import Agents.ForwardModel as ForwardModel
import Environments.BaseEnv as BaseEnv
import Utils.SharedCoreTypes as SCT
import Utils.UserInputHelper as UI
from DataManager.DataManager import DataManager
from Utils.Metrics.Logger import Logger
from Utils.PathHelper import GetRootPath
import Utils.ConfigHelper as ConfigHelper


class Runner:

	def __init__(self,
			configPath:str,
			runPath:str,
			env:BaseEnv.BaseEnv,
			agents:typing.List[BaseAgent.BaseAgent],
			load:bool,
			forwardModel:ForwardModel.ForwardModel,
			maxEpisodesOverride:typing.Optional[int] = None,
			maxStepsOverride:typing.Optional[int] = None):

		self.ConfigPath = configPath
		self._RunPath = runPath
		self.Env = env
		self._DataManager = DataManager()
		self._Logger = Logger()
		self.Agents = agents

		self._MaxEpisodesOverride = maxEpisodesOverride
		self._MaxStepsOverride = maxStepsOverride

		self.ForwardModel = forwardModel
		self.LoadConfig()

		if load:
			self.Load()
		return

	def LoadConfig(self) -> None:

		self.Config = ConfigHelper.LoadConfig(self.ConfigPath)

		self._DataManager.LoadConfig(self.Config)

		self.Env.LoadConfig(self.Config)

		agentConfig = self.Config.get("AgentConfig", {})
		for agent in self.Agents:
			agent.LoadConfig(agentConfig)

		return



	def RunEpisodes(self) -> None:
		lastRewards:typing.Deque[float] = deque(maxlen=10)
		lastTimes:typing.Deque[float] = deque(maxlen=10)

		episode = 0
		while episode < self._GetMaxEpisodes():
			startTime = time.process_time()

			steps, reward = self.RunEpisode()

			timeTaken = time.process_time() - startTime

			lastRewards.append(reward)
			lastTimes.append(timeTaken/steps)

			avgReward = sum(lastRewards) / len(lastRewards)
			avgTime = sum(lastTimes) / len(lastTimes)

			print(f"Episode:{episode+1} steps:{steps+1} reward:{reward} avg:{avgReward} time:{timeTaken:.2f} avg:{avgTime:.2f}")

			episode += 1

		self.Save()
		return

	def RunEpisode(self) -> typing.Tuple[int, float]:


		totalReward:float = 0.0
		state = self.Env.Reset()
		for step in range(self._GetMaxSteps()):

			with self._Logger.Time("Step"):

				with self._Logger.Time("GetAction"):
					action = self.GetAction(state)

				with self._Logger.Time("Step"):
					nextState, reward, terminated, truncated = self.Env.Step(action)
					truncated = truncated or step >= self._GetMaxSteps() - 1

				with self._Logger.Time("Remember"):
					self.Remember(state, action, reward, nextState, terminated, truncated)

				totalReward += reward

				self._Logger.StepEnd(reward, terminated, truncated)


				# check if user wants to reload config
				if UI.IsKeyPressed('alt+c'):
					self.LoadConfig()
					print("+++++++ Loaded Config +++++++")

				if UI.IsKeyPressed('alt+s'):
					self.Save()
					print("+++++++ Saved Agent +++++++")

				if UI.IsKeyPressed('alt+l'):
					self.Load()
					print("+++++++ Loaded Agent +++++++")

				state = nextState
				if terminated or truncated:
					break

		with self._Logger.Time("Reset"):
			self.Reset()
		return step, totalReward

	def GetAction(self, state:SCT.State) -> SCT.Action:
		return self.Agents[0].GetAction(state)

	def Remember(self,
			state:SCT.State,
			action:SCT.Action,
			reward:SCT.Reward,
			nextState:SCT.State,
			terminated:bool,
			truncated:bool) -> None:
		with self._Logger.Time("DataManager"):
			self._DataManager.EnvRemember(state, action, reward, nextState, terminated, truncated)

		with self._Logger.Time("Agents"):
			for agent in self.Agents:
				agent.Remember(state, action, reward, nextState, terminated, truncated)

		with self._Logger.Time("ForwardModel"):
			self.ForwardModel.Remember(state, action, reward, nextState, terminated, truncated)
		return

	def Reset(self) -> None:
		with self._Logger.Time("DataManager"):
			self._DataManager.EnvReset()

		with self._Logger.Time("Agents"):
			for agent in self.Agents:
				agent.Reset()
		return

	def Save(self) -> None:
		path = os.path.join(GetRootPath(), "Data", self.Config["Name"])
		if not os.path.exists(path):
			os.makedirs(path)

		self._DataManager.Save(path)

		for agent in self.Agents:
			agent.Save(path)
		return

	def Load(self) -> None:
		path = os.path.join(GetRootPath(), "Data", self.Config["Name"])
		if not os.path.exists(path):
			return

		self._DataManager.Load(path)

		for agent in self.Agents:
			agent.Load(path)
		return


	def _GetMaxEpisodes(self) -> int:
		if self._MaxEpisodesOverride is not None:
			return self._MaxEpisodesOverride
		return self.Config["MaxEpisodes"]

	def _GetMaxSteps(self) -> int:
		if self._MaxStepsOverride is not None:
			return self._MaxStepsOverride
		return self.Config["MaxSteps"]
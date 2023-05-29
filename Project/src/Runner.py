import json
import os
import time
import typing
from collections import deque

import src.Agents.BaseAgent as BaseAgent
import src.Agents.ForwardModel as ForwardModel
import src.DataManager.DataColumnTypes as DCT
import src.Environments.BaseEnv as BaseEnv
import src.Utils.SharedCoreTypes as SCT
import src.Utils.UserInputHelper as UI
from src.DataManager.DataManager import DataManager
from src.Utils.Metrics.Logger import Logger
from src.Utils.PathHelper import GetRootPath


class Runner:

	def __init__(self,
			configPath:str,
			runPath:str,
			env:BaseEnv.BaseEnv,
			agents:typing.List[BaseAgent.BaseAgent],
			load:bool,
			forwardModel:ForwardModel.ForwardModel):

		self.ConfigPath = configPath
		self._RunPath = runPath
		self.Env = env
		self._DataManager = DataManager()
		self._Logger = Logger()
		self.Agents = agents

		self.ForwardModel = forwardModel
		self.LoadConfig()

		if load:
			self.Load()
		return

	def LoadConfig(self) -> None:

		# load environment config
		with open(self.ConfigPath) as f:
			self.Config = json.load(f)

		self._DataManager.LoadConfig(self.Config)

		self.Env.LoadConfig(self.Config)

		for agent in self.Agents:
			agent.LoadConfig(self.Config)

		return



	def RunEpisodes(self) -> None:
		lastRewards:typing.Deque[float] = deque(maxlen=10)
		lastTimes:typing.Deque[float] = deque(maxlen=10)

		episode = 0
		while episode < self.Config["MaxEpisodes"]:
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
		for step in range(self.Config["MaxSteps"]):

			with self._Logger.Time("Step"):

				with self._Logger.Time("GetAction"):
					action = self.GetAction(state)

				with self._Logger.Time("Step"):
					nextState, reward, terminated, truncated = self.Env.Step(action)
					truncated = truncated or step >= self.Config["MaxSteps"] - 1

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
		path = os.path.join(GetRootPath(), "data", self.Config["Name"])
		if not os.path.exists(path):
			os.makedirs(path)

		self._DataManager.Save(path)

		for agent in self.Agents:
			agent.Save(path)
		return

	def Load(self) -> None:
		path = os.path.join(GetRootPath(), "data", self.Config["Name"])
		if not os.path.exists(path):
			return

		self._DataManager.Load(path)

		for agent in self.Agents:
			agent.Load(path)
		return

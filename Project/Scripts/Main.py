import Utils.SharedCoreTypes as SCT

import json
import os

import Agents.BaseAgent as BaseAgent
import Environments.BaseEnv as BaseEnv
import keyboard
import Utils.UserInputHelper as UI
from DataManager.DataManager import DataManager
from Utils.PathHelper import GetRootPath


class Runner:

	def __init__(self, configPath:str, env:BaseEnv.BaseEnv, agents:list[BaseAgent.BaseAgent], load:bool):
		self.ConfigPath = configPath
		self.Env = env
		self._DataManager = DataManager()
		self.Agents = agents

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
		lastRewards = []

		episode = 0
		while episode < self.Config["MaxEpisodes"]:
			step, reward = self.RunEpisode()

			lastRewards.append(reward)
			if len(lastRewards) > 10:
				lastRewards.pop(0)

			avgReward = sum(lastRewards) / len(lastRewards)

			print(f"Episode:{episode+1} steps:{step+1} reward:{reward} avg:{avgReward}")

			episode += 1

		self.Save()
		return

	def RunEpisode(self) -> tuple[int, float]:

		totalReward:float = 0.0
		state = self.Env.Reset()
		for step in range(self.Config["MaxSteps"]):

			action = self.GetAction(state)

			nextState, reward, terminated, truncated = self.Env.Step(action)
			truncated = truncated or step >= self.Config["MaxSteps"] - 1

			self.Remember(state, action, reward, nextState, terminated, truncated)

			totalReward += reward


			# check if user wants to reload config
			if keyboard.is_pressed('alt+c'):
				self.LoadConfig()
				print("+++++++ Loaded Config +++++++")

			if keyboard.is_pressed('alt+s'):
				self.Save()
				print("+++++++ Saved Agent +++++++")

			if keyboard.is_pressed('alt+l'):
				self.Load()
				print("+++++++ Loaded Agent +++++++")

			state = nextState
			if terminated or truncated:
				break

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

		# update data manager
		self._DataManager.EnvRemember(state, action, reward, nextState, terminated, truncated)

		# update agents
		for agent in self.Agents:
			agent.Remember(state, action, reward, nextState, terminated, truncated)
		return

	def Reset(self) -> None:
		# update data manager
		self._DataManager.EnvReset()

		# update agents
		for agent in self.Agents:
			agent.Reset()
		return

	def Save(self) -> None:
		path = os.path.join(GetRootPath(), "data", self.Config["Name"])
		if not os.path.exists(path):
			os.makedirs(path)

		for agent in self.Agents:
			agent.Save(path)
		return

	def Load(self) -> None:
		path = os.path.join("data", self.Config["Name"])
		if not os.path.exists(path):
			return

		for agent in self.Agents:
			agent.Load(path)
		return




def Main() -> None:
	# find all environments in the configs folder
	configPath = os.path.join(GetRootPath(), "Config", "Envs")
	envConfigPath = UI.FilePicker("Environments", configPath)

	# load config
	with open(envConfigPath) as f:
		config = json.load(f)

	# load env
	env = BaseEnv.GetEnv(config)

	# load data manager
	dataManager = DataManager()
	dataManager.Setup(config, env)

	# load agents
	numAgents = UI.NumPicker("Number of agents", 1, 1)

	mode = BaseAgent.AgentMode.Train
	if UI.BoolPicker("Play"):
		mode = BaseAgent.AgentMode.Play

	load = UI.BoolPicker("Load")

	agents = []
	for i in range(numAgents):
		agentType = UI.OptionPicker(f"Agent_{i+1}", BaseAgent.AgentList)
		agent = BaseAgent.GetAgent(agentType)(env, config, mode=mode)
		agents.append(agent)

	# run
	runner = Runner(envConfigPath, env, agents, load)
	runner.RunEpisodes()

	return


if __name__ == "__main__":

	try:

		Main()

	except KeyboardInterrupt:
		print("")
		print("Interrupted")
		os._exit(0)

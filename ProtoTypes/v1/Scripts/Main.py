import keyboard
import json
import Utils.UserInputHelper as UI
from Utils.PathHelper import GetRootPath
from Agents import BaseAgent
from Environments import BaseEnv
import os

class Runner:

	def __init__(self, configPath):
		self.ConfigPath = configPath
		self.Config = None
		self.Env = None
		self.Agents = None

		self.LoadConfig()
		return

	def LoadConfig(self):

		# load environment config
		with open(self.ConfigPath) as f:
			self.Config = json.load(f)

		# ensure environment loaded
		if self.Env is None:
			self.Env = BaseEnv.GetEnv(self.Config)


		# ensure agents loaded
		if self.Agents is None:

			numAgents = UI.NumPicker("Number of agents", 1, 1)
			agentOptions = BaseAgent.AgentList

			self.Agents = []
			for i in range(numAgents):
				agentType = UI.OptionPicker(f"Agent_{i+1}", agentOptions)
				agent = BaseAgent.GetAgent(agentType)(self.Env, self.Config)

				self.Agents.append(agent)

			if UI.BoolPicker("Load"):
				self.Load()
		else:
			for agent in self.Agents:
				agent.LoadConfig(self.Config)

		return



	def RunEpisodes(self, numEpisodes=1):
		lastRewards = []
		for episode in range(numEpisodes):
			step, reward = self.RunEpisode()
			lastRewards.append(reward)
			if len(lastRewards) > 10:
				lastRewards.pop(0)
			avgReward = sum(lastRewards) / len(lastRewards)
			print(f"Episode:{episode} steps:{step+1} reward:{reward} avg:{avgReward}")

		self.Save()
		return

	def RunEpisode(self):

		totalReward = 0
		state = self.Env.Reset()
		for step in range(self.Config["MaxSteps"]):

			action = self.GetAction(state)

			# print(f"state: {state}, action: {action}")

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



		# update agents
		for agent in self.Agents:
			agent.Reset()
		return step, totalReward



	def GetAction(self, observation):
		return self.Agents[0].GetAction(observation)

	def Remember(self, state, action, reward, nextState, terminated, truncated):
		for agent in self.Agents:
			agent.Remember(state, action, reward, nextState, terminated, truncated)
		return

	def Save(self):
		path = os.path.join(GetRootPath(), "data", self.Config["Name"])
		if not os.path.exists(path):
			os.makedirs(path)

		for agent in self.Agents:
			agent.Save(path)
		return

	def Load(self):
		path = os.path.join("data", self.Config["Name"])
		if not os.path.exists(path):
			return

		for agent in self.Agents:
			agent.Load(path)
		return




def Main():
	# find all environments in the configs folder
	configPath = os.path.join(GetRootPath(), "Config", "Envs")

	envConfigPath = UI.FilePicker("Environments", configPath)

	runner = Runner(envConfigPath)

	runner.RunEpisodes(numEpisodes=1000)

	return


if __name__ == "__main__":

	try:

		Main()

	except KeyboardInterrupt:
		print("")
		print("Interrupted")
		os._exit(0)

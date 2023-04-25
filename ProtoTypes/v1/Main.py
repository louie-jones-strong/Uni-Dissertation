import keyboard
import json
import gym
import Utils.UserInputHelper as UI
from Agents import BaseAgent
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
			kargs = self.Config.get("kwargs", {})
			self.Env = gym.make(self.Config["GymID"], render_mode=self.Config["RenderMode"], **kargs)
			print(self.Env.metadata)
			self.Env.metadata["render_fps"] = 100_000

		# ensure agents loaded
		if self.Agents is None:

			numAgents = UI.NumPicker("Number of agents", 1, 1)
			agentOptions = ["Random", "DQN", "Human"]

			self.Agents = []
			for i in range(numAgents):
				agentType = UI.OptionPicker(f"Agent_{i+1}", agentOptions)
				agent = BaseAgent.GetAgent(agentType)(self.Env, self.Config)

				if agentType == "DQN" and UI.BoolPicker("Load"):
					path = os.path.join("data", self.Config["Name"])
					agent.Load(path)

				self.Agents.append(agent)
		else:
			for agent in self.Agents:
				agent.LoadConfig(self.Config)

		return



	def RunEpisodes(self, numEpisodes=1):
		lastRewards = []
		for episode in range(numEpisodes):
			reward = self.RunEpisode()
			lastRewards.append(reward)
			if len(lastRewards) > 10:
				lastRewards.pop(0)
			avgReward = sum(lastRewards) / len(lastRewards)
			print(f"Episode: {episode}, reward: {reward}, avg reward: {avgReward}")

		return

	def RunEpisode(self):

		totalReward = 0
		state, info = self.Env.reset()
		for step in range(self.Config["MaxSteps"]):

			action = self.GetAction(state)

			nextState, reward, terminated, truncated, info = self.Env.step(action)

			done = terminated or truncated or step >= self.Config["MaxSteps"] - 1

			self.Remember(state, action, reward, nextState, done)

			totalReward += reward

			# check if user wants to stop
			if keyboard.is_pressed('ctrl+q'):
				raise KeyboardInterrupt

			# check if user wants to reload config
			if keyboard.is_pressed('ctrl+c'):
				self.LoadConfig()

			if keyboard.is_pressed('ctrl+s'):
				self.Save()

			if keyboard.is_pressed('ctrl+l'):
				self.Load()

			state = nextState
			if done:
				break



		# update agents
		for agent in self.Agents:
			agent.Reset()
		return totalReward



	def GetAction(self, observation):
		return self.Agents[0].GetAction(observation)

	def Remember(self, state, action, reward, nextState, done):
		for agent in self.Agents:
			agent.Remember(state, action, reward, nextState, done)
		return

	def Save(self):
		path = os.path.join("data", self.Config["Name"])
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
	configPath = os.path.join(os.path.abspath(os.curdir), "Config", "Envs")

	envConfigPath = UI.FilePicker("Environments", configPath)

	runner = Runner(envConfigPath)

	try:
		runner.RunEpisodes(numEpisodes=1000)
	except KeyboardInterrupt:
		print('Interrupted')
		os._exit(0)

	return


if __name__ == "__main__":
	Main()

import keyboard
import json
import gym
import Utils.UserInputHelper as UI
from Agents import BaseAgent

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
			self.Env = gym.make(self.Config["GymID"], render_mode=self.Config["RenderMode"])

		# ensure agents loaded
		if self.Agents is None:

			numAgents = UI.NumPicker("Number of agents", 1, 1)
			agentOptions = ["Random", "DQN", "Human"]

			self.Agents = []
			for i in range(numAgents):
				agentType = UI.OptionPicker(f"Agent_{i+1}", agentOptions)
				agent = BaseAgent.GetAgent(agentType)(self.Env, self.Config)

				self.Agents.append(agent)
		else:
			for agent in self.Agents:
				agent.Config = agent.LoadConfig(self.Config)

		return



	def RunEpisodes(self, numEpisodes=1, maxSteps=1000):
		lastRewards = []
		for episode in range(numEpisodes):
			reward = self.RunEpisode(maxSteps=maxSteps)
			lastRewards.append(reward)
			if len(lastRewards) > 10:
				lastRewards.pop(0)
			avgReward = sum(lastRewards) / len(lastRewards)
			print(f"Episode: {episode}, reward: {reward}, avg reward: {avgReward}")

		return

	def RunEpisode(self, maxSteps=1000):

		totalReward = 0
		state, info = self.Env.reset()
		for step in range(maxSteps):

			action = self.GetAction(state)

			nextState, reward, terminated, truncated, info = self.Env.step(action)

			done = terminated or truncated

			self.Remember(state, action, reward, nextState, done)

			totalReward += reward

			# check if user wants to stop
			if keyboard.is_pressed('ctrl+c'):
				raise KeyboardInterrupt

			# check if user wants to reload config
			if keyboard.is_pressed('ctrl+l'):
				self.LoadConfig()


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
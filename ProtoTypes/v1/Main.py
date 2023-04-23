import gym
from Agents import RandomAgent, DQNAgent





class Runner:

	def __init__(self, env, agents):

		self.Env = env
		self.Agents = agents
		return

	def RunEpisodes(self, numEpisodes=1, maxSteps=1000):
		for episode in range(numEpisodes):
			self.RunEpisode(maxSteps=maxSteps)
		return

	def RunEpisode(self, maxSteps=1000):
		replayBuffer = []

		observation, info = self.Env.reset()
		for step in range(maxSteps):

			action = self.GetAction(observation)

			observation, reward, terminated, truncated, info = self.Env.step(action)

			replayBuffer.append((observation, reward, terminated, truncated, info))


			if terminated or truncated:
				break

		# update agents with replay buffer
		for agent in self.Agents:
			agent.Reset()

		replayBuffer.clear()

		return

	def GetAction(self, observation):
		return self.Agents[0].GetAction(observation)

	def __del__(self):
		self.Env.close()
		return






env = gym.make("CartPole-v1", render_mode="human")
agents = []
# agents += [RandomAgent.RandomAgent(env)]
agents += [DQNAgent.DQNAgent(env)]

runner = Runner(env, agents)
runner.RunEpisodes(numEpisodes=1, maxSteps=1000)
runner.RunEpisodes(numEpisodes=1, maxSteps=1000)
runner.RunEpisodes(numEpisodes=1, maxSteps=1000)
del runner
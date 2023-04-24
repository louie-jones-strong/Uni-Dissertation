import gym
from Agents import RandomAgent, DQNAgent
import Utils.ReplayBuffer as ReplayBuffer
import os

class Runner:

	def __init__(self, env, replayBuffer, agents):

		self.Env = env
		self.Agents = agents
		self.ReplayBuffer = replayBuffer
		self.TransitionAcc = ReplayBuffer.TransitionAccumulator(1000)
		return

	def RunEpisodes(self, numEpisodes=1, maxSteps=1000):
		for episode in range(numEpisodes):
			self.RunEpisode(maxSteps=maxSteps)

		return

	def RunEpisode(self, maxSteps=1000):
		self.TransitionAcc.Clear()

		state, info = self.Env.reset()
		for step in range(maxSteps):

			action = self.GetAction(state)

			nextState, reward, terminated, truncated, info = self.Env.step(action)

			done = terminated or truncated
			self.TransitionAcc.Add(state, action, reward, nextState, done)

			state = nextState
			if done:
				break


		self.TransitionAcc.TransferToReplayBuffer(self.ReplayBuffer)

		# update agents with replay buffer
		for agent in self.Agents:
			agent.Reset()
		return

	def GetAction(self, observation):
		return self.Agents[0].GetAction(observation)

	def __del__(self):
		self.Env.close()
		return




try:
	env = gym.make("CartPole-v1", render_mode="human")
	replayBuffer = ReplayBuffer.ReplayBuffer(1_000_000)

	agents = []
	# agents.append(RandomAgent.RandomAgent(env, replayBuffer))
	agents.append(DQNAgent.DQNAgent(env, replayBuffer))


	runner = Runner(env, replayBuffer, agents)
	runner.RunEpisodes(numEpisodes=1000, maxSteps=1000)
	print(f"Replay buffer size: {len(replayBuffer)}")
	del runner
except KeyboardInterrupt:
	print('Interrupted')
	os._exit(0)
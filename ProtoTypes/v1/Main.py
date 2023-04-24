import gym
from Agents import RandomAgent, DQNAgent
import Utils.ReplayBuffer as ReplayBuffer
import os
import keyboard

class Runner:

	def __init__(self, env, replayBuffer, agents):

		self.Env = env
		self.Agents = agents
		self.ReplayBuffer = replayBuffer
		self.TransitionAcc = ReplayBuffer.TransitionAccumulator(1000)
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
		# self.TransitionAcc.Clear()

		totalReward = 0
		state, info = self.Env.reset()
		for step in range(maxSteps):

			action = self.GetAction(state)

			nextState, reward, terminated, truncated, info = self.Env.step(action)

			done = terminated or truncated
			# self.TransitionAcc.Add(state, action, reward, nextState, done)
			self.ReplayBuffer.Add(state, action, reward, nextState, done)

			totalReward += reward

			# check if user wants to stop ctrl+c
			if keyboard.is_pressed('ctrl+c'):
				raise KeyboardInterrupt

			if keyboard.is_pressed('ctrl+l'):
				self.ReloadConfig()


			state = nextState
			if done:
				break


		# self.TransitionAcc.TransferToReplayBuffer(self.ReplayBuffer)

		# update agents with replay buffer
		for agent in self.Agents:
			agent.Reset()
		return totalReward

	def GetAction(self, observation):
		return self.Agents[0].GetAction(observation)

	def ReloadConfig(self):
		for agent in self.Agents:
			agent.Config = agent.LoadConfig()
		return

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
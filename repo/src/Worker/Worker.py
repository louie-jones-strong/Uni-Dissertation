import typing
import reverb

# import src.Common.Agents.BaseAgent as BaseAgent
import src.Common.Environments.BaseEnv as BaseEnv
import src.Common.Utils.ConfigHelper as ConfigHelper
import src.Common.Utils.UserInputHelper as UI
import os
from src.Common.Utils.PathHelper import GetRootPath
from collections import deque



class Runner:

	def __init__(self, env, maxSteps, client) -> None:
		self.Env = env
		self.MaxSteps = maxSteps
		self.Client = client
		self.TragetoryStepCount = 1

		self.State = self.Env.Reset()
		self.StepCount = 0
		self.TotalReward = 0


		self.TransitionBuffer = deque()
		return

	def GetState(self):
		return self.State

	def Step(self, action):

		nextState, reward, terminated, truncated = self.Env.Step(action)

		truncated = truncated or self.StepCount >= self.MaxSteps
		self.StepCount += 1

		self.TransitionBuffer.append((self.State, action, reward, nextState, terminated, truncated))
		self.TotalReward += reward

		self.State = nextState

		if terminated or truncated:
			self.Reset()

		return nextState, terminated or truncated


	def Reset(self):


		# empty the transition buffer
		numTransitions = len(self.TransitionBuffer)
		with self.Client.trajectory_writer(num_keep_alive_refs=numTransitions) as writer:
			for i in range(numTransitions):
				transition = self.TransitionBuffer.pop()
				state, action, reward, nextState, terminated, truncated = transition
				self.TotalReward -= reward

				writer.append({
					"State": state,
					"NextState": nextState,
					"Action": action,
					"Reward": reward,
					"FutureReward": self.TotalReward,
					"Terminated": terminated,
					"Truncated": truncated
				})

				if i >= self.TragetoryStepCount:

					writer.create_item(
						table="Trajectories",
						priority=1.5,
						trajectory={
							"State": writer.history["State"][-self.TragetoryStepCount:],
							"NextState": writer.history["NextState"][-self.TragetoryStepCount:],
							"Action": writer.history["Action"][-self.TragetoryStepCount:],
							"Reward": writer.history["Reward"][-self.TragetoryStepCount:],
							"FutureReward": writer.history["FutureReward"][-self.TragetoryStepCount:],
							"Terminated": writer.history["Terminated"][-self.TragetoryStepCount:],
							"Truncated": writer.history["Truncated"][-self.TragetoryStepCount:],
						})

			# This call blocks until all the items (in this case only one) have been
			# sent to the server, inserted into respective tables and confirmations
			# received by the writer.
			writer.end_episode(timeout_ms=1000)

			# Ending the episode also clears the history property which is why we are
			# able to use `[:]` in when defining the trajectory above.
			assert len(writer.history["State"]) == 0
			assert len(writer.history["NextState"]) == 0
			assert len(writer.history["Action"]) == 0
			assert len(writer.history["Reward"]) == 0
			assert len(writer.history["FutureReward"]) == 0
			assert len(writer.history["Terminated"]) == 0
			assert len(writer.history["Truncated"]) == 0

			assert len(self.TransitionBuffer) == 0

		assert abs(self.TotalReward) <= 0.000_0001, f"TotalReward:{self.TotalReward}"

		self.State = self.Env.Reset()
		self.StepCount = 0
		self.TotalReward = 0
		return







class Worker:

	def __init__(self, envConfig, numEnvs:int):#, agents:typing.List[BaseAgent.BaseAgent]):

		self.Config = envConfig
		# self.Agents = agents

		client = reverb.Client(f'experience-store:{5001}')


		self.Envs = []
		for i in range(numEnvs):
			env = BaseEnv.GetEnv(self.Config )

			runnner = Runner(env, self.Config ["MaxSteps"], client)
			self.Envs.append(runnner)

			self.ActionSpace = env.ActionSpace


		self.EpisodeCount = 0
		return

	def Run(self) -> None:

		stateList = [env.GetState() for env in self.Envs]
		while self.EpisodeCount < self.Config["MaxEpisodes"]:

			# get agent action
			actions = self.GetActions(stateList)

			# step the envirements
			self.StepEnvs(actions)


		return

	def GetActions(self, stateList):
		actions = []
		for i in range(len(stateList)):
			# action = self.Agents[0].GetAction(stateList[i])
			action = self.ActionSpace.sample()
			actions.append(action)
		return actions

	def StepEnvs(self, actions):

		stateList = []
		for i in range(len(self.Envs)):

			state, done = self.Envs[i].Step(actions[i])
			if done:
				maxEpisodes = self.Config["MaxEpisodes"]
				print(f"{self.EpisodeCount+1} / {maxEpisodes}")
				self.EpisodeCount += 1

			stateList.append(state)

		return stateList



def Run(envConfig) -> None:
	numEnvs = 1
	numAgents = 1

	agents = []

	# for i in range(agents):
	# 	agent = RandomAgent.RandomAgent({}, BaseAgent.AgentMode.Play)
	# 	agents.append(agent)


	worker = Worker(envConfig, numEnvs)#, agents)
	worker.Run()
	return
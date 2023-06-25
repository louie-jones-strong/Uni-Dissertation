from collections import deque
import src.Worker.Environments.BaseEnv as BaseEnv
import src.Common.Utils.Metrics.Logger as Logger

class EnvRunner:

	def __init__(self, env:BaseEnv.BaseEnv, maxSteps, experienceStore) -> None:
		self.Env = env
		self.MaxSteps = maxSteps
		self.ExperienceStore = experienceStore
		self.TragetoryStepCount = 1

		self.State = self.Env.Reset()
		self.StepCount = 0
		self.TotalReward = 0


		self.TransitionBuffer = deque()
		self._Logger = Logger.Logger()
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

			# log the episode
			self._Logger.LogDict({
				"Terminated": float(terminated),
				"Truncated": float(truncated),
				"EpisodeTotalReward": self.TotalReward,
				"EpisodeSteps": self.StepCount},
				commit=True)


			self.Reset()

		return nextState, terminated or truncated


	def Reset(self):


		# empty the transition buffer
		numTransitions = len(self.TransitionBuffer)
		with self.ExperienceStore.trajectory_writer(num_keep_alive_refs=numTransitions) as writer:
			for i in range(numTransitions):
				transition = self.TransitionBuffer.pop()
				state, action, reward, nextState, terminated, truncated = transition
				self.TotalReward -= reward # todo discount factor

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


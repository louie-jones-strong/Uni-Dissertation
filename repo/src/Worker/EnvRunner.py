from collections import deque
import src.Worker.Environments.BaseEnv as BaseEnv
import src.Common.Utils.Metrics.Logger as Logger
import src.Common.Enums.DataColumnTypes as DCT

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
					DCT.DataColumnTypes.CurrentState.name: state,
					DCT.DataColumnTypes.NextState.name: nextState,
					DCT.DataColumnTypes.Action.name: action,
					DCT.DataColumnTypes.Reward.name: reward,
					DCT.DataColumnTypes.MaxFutureRewards.name: self.TotalReward,
					DCT.DataColumnTypes.Terminated.name: terminated,
					DCT.DataColumnTypes.Truncated.name: truncated
				})

				if i >= self.TragetoryStepCount:

					writer.create_item(
						table="Trajectories",
						priority=1.5,
						trajectory={
							DCT.DataColumnTypes.CurrentState.name: writer.history[DCT.DataColumnTypes.CurrentState.name][-self.TragetoryStepCount:],
							DCT.DataColumnTypes.NextState.name: writer.history[DCT.DataColumnTypes.NextState.name][-self.TragetoryStepCount:],
							DCT.DataColumnTypes.Action.name: writer.history[DCT.DataColumnTypes.Action.name][-self.TragetoryStepCount:],
							DCT.DataColumnTypes.Reward.name: writer.history[DCT.DataColumnTypes.Reward.name][-self.TragetoryStepCount:],
							DCT.DataColumnTypes.MaxFutureRewards.name: writer.history[DCT.DataColumnTypes.MaxFutureRewards.name][-self.TragetoryStepCount:],
							DCT.DataColumnTypes.Terminated.name: writer.history[DCT.DataColumnTypes.Terminated.name][-self.TragetoryStepCount:],
							DCT.DataColumnTypes.Truncated.name: writer.history[DCT.DataColumnTypes.Truncated.name][-self.TragetoryStepCount:],
						})

			# This call blocks until all the items (in this case only one) have been
			# sent to the server, inserted into respective tables and confirmations
			# received by the writer.
			writer.end_episode(timeout_ms=1000)

			# Ending the episode also clears the history property which is why we are
			# able to use `[:]` in when defining the trajectory above.
			assert len(writer.history[DCT.DataColumnTypes.CurrentState.name]) == 0
			assert len(writer.history[DCT.DataColumnTypes.NextState.name]) == 0
			assert len(writer.history[DCT.DataColumnTypes.Action.name]) == 0
			assert len(writer.history[DCT.DataColumnTypes.Reward.name]) == 0
			assert len(writer.history[DCT.DataColumnTypes.MaxFutureRewards.name]) == 0
			assert len(writer.history[DCT.DataColumnTypes.Terminated.name]) == 0
			assert len(writer.history[DCT.DataColumnTypes.Truncated.name]) == 0

			assert len(self.TransitionBuffer) == 0

		assert abs(self.TotalReward) <= 0.000_0001, f"TotalReward:{self.TotalReward}"

		self.State = self.Env.Reset()
		self.StepCount = 0
		self.TotalReward = 0
		return


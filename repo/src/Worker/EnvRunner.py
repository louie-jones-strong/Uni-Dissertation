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
		currentState_Name = DCT.DataColumnTypes.CurrentState.name
		nextState_Name = DCT.DataColumnTypes.NextState.name
		action_Name = DCT.DataColumnTypes.Action.name
		reward_Name = DCT.DataColumnTypes.Reward.name
		maxFutureRewards_Name = DCT.DataColumnTypes.MaxFutureRewards.name
		terminated_Name = DCT.DataColumnTypes.Terminated.name
		truncated_Name = DCT.DataColumnTypes.Truncated.name

		# empty the transition buffer
		numTransitions = len(self.TransitionBuffer)
		with self.ExperienceStore.trajectory_writer(num_keep_alive_refs=numTransitions) as writer:
			for i in range(numTransitions):
				transition = self.TransitionBuffer.pop()
				state, action, reward, nextState, terminated, truncated = transition
				self.TotalReward -= reward  # todo discount factor

				writer.append({
					currentState_Name: state,
					nextState_Name: nextState,
					action_Name: action,
					reward_Name: reward,
					maxFutureRewards_Name: self.TotalReward,
					terminated_Name: terminated,
					truncated_Name: truncated
				})

				if i >= self.TragetoryStepCount:

					writer.create_item(
						table="Trajectories",
						priority=1.5,
						trajectory={
							currentState_Name: writer.history[currentState_Name][-self.TragetoryStepCount:],
							nextState_Name: writer.history[nextState_Name][-self.TragetoryStepCount:],
							action_Name: writer.history[action_Name][-self.TragetoryStepCount:],
							reward_Name: writer.history[reward_Name][-self.TragetoryStepCount:],
							maxFutureRewards_Name: writer.history[maxFutureRewards_Name][-self.TragetoryStepCount:],
							terminated_Name: writer.history[terminated_Name][-self.TragetoryStepCount:],
							truncated_Name: writer.history[truncated_Name][-self.TragetoryStepCount:],
						})

			# This call blocks until all the items (in this case only one) have been
			# sent to the server, inserted into respective tables and confirmations
			# received by the writer.
			writer.end_episode(timeout_ms=1000)

			# Ending the episode also clears the history property which is why we are
			# able to use `[:]` in when defining the trajectory above.
			assert len(writer.history[currentState_Name]) == 0
			assert len(writer.history[nextState_Name]) == 0
			assert len(writer.history[action_Name]) == 0
			assert len(writer.history[reward_Name]) == 0
			assert len(writer.history[maxFutureRewards_Name]) == 0
			assert len(writer.history[terminated_Name]) == 0
			assert len(writer.history[truncated_Name]) == 0

			assert len(self.TransitionBuffer) == 0

		assert abs(self.TotalReward) <= 0.000_0001, f"TotalReward:{self.TotalReward}"

		self.State = self.Env.Reset()
		self.StepCount = 0
		self.TotalReward = 0
		return


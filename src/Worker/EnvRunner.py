import src.Worker.Environments.BaseEnv as BaseEnv
import src.Common.Utils.Metrics.Logger as Logger

class EnvRunner:

	def __init__(self, env:BaseEnv.BaseEnv, maxSteps, experienceStore) -> None:
		self.Env = env
		self.MaxSteps = maxSteps
		self.ExperienceStore = experienceStore

		self.State = self.Env.Reset()
		self.StepCount = 0
		self.TotalReward = 0


		self._Logger = Logger.Logger()
		return

	def GetState(self):
		return self.State

	def Step(self, action):

		nextState, reward, terminated, truncated = self.Env.Step(action)

		truncated = truncated or self.StepCount >= self.MaxSteps
		self.StepCount += 1

		self.ExperienceStore.AddTransition(self.State, nextState, action, reward, terminated, truncated)

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


	def Reset(self) -> None:
		self.ExperienceStore.EmptyTransitionBuffer()
		self.State = self.Env.Reset()
		self.StepCount = 0
		self.TotalReward = 0
		return


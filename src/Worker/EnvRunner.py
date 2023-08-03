import src.Worker.Environments.BaseEnv as BaseEnv
import src.Common.Utils.Metrics.Logger as Logger
from src.Common.EpisodeReplay.EpisodeReplay import EpisodeReplay as ER
from src.Common.EpisodeReplay.EpisodeReplayStep import EpisodeReplayStep as ERStep

class EnvRunner:

	def __init__(self, env:BaseEnv.BaseEnv, maxSteps:int, experienceStore, replayFolder=None) -> None:
		self.Env = env
		self.MaxSteps = maxSteps
		self.ExperienceStore = experienceStore
		self.ReplayFolder = replayFolder

		self.State = self.Env.Reset()
		self.StepCount = 0
		self.TotalReward = 0
		self.EpisodeReplay = None


		self._Logger = Logger.Logger()
		return

	def GetState(self):
		return self.State

	def Step(self, action, actionReason=None):

		if self.EpisodeReplay is not None:

			frameNum = self.Env._CurrentStep
			humanState = self.Env.Render()
			agentState = self.State

			nextState, reward, terminated, truncated = self.Env.Step(action)

			erStep = ERStep(frameNum, humanState, agentState, reward, action, actionReason)
			self.EpisodeReplay.AddStep(erStep)

		else:
			nextState, reward, terminated, truncated = self.Env.Step(action)



		truncated = truncated or self.StepCount >= self.MaxSteps
		self.StepCount += 1

		self.ExperienceStore.AddTransition(self.State, nextState, action, reward, terminated, truncated)

		self.TotalReward += reward

		self.State = nextState

		if terminated or truncated:

			if self.EpisodeReplay is not None:
				self.EpisodeReplay.EpisodeEnd(terminated, truncated)
				self.EpisodeReplay.SaveToFolder(self.ReplayFolder)

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

		if self.ReplayFolder is not None:
			self.EpisodeReplay = ER()
		return


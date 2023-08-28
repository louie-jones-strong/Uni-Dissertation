import src.Worker.Environments.BaseEnv as BaseEnv
import src.Common.Utils.Metrics.Logger as Logger
from src.Common.EpisodeReplay.EpisodeReplay import EpisodeReplay as ER
from src.Common.EpisodeReplay.EpisodeReplayStep import EpisodeReplayStep as ERStep
import src.Common.Utils.Config.ConfigHelper as ConfigHelper
import src.Common.Utils.PathHelper as PathHelper
import os

class EnvRunner:

	def __init__(self, env:BaseEnv.BaseEnv, maxSteps:int, experienceStore,
			replayFolder=None, replayInfo=None, humanRender=True) -> None:

		self.Env = env
		self.MaxSteps = maxSteps
		self.ExperienceStore = experienceStore
		self.ReplayFolder = replayFolder
		self.ReplayInfo = replayInfo
		self.HumanRender = humanRender

		self.State = self.Env.Reset()
		self.StepCount = 0
		self.TotalReward = 0
		self.EpisodeReplay = None


		self._Logger = Logger.Logger()
		return

	def GetState(self):
		return self.State

	def Step(self, action, actionReason=None):

		if self.EpisodeReplay is None:
			nextState, reward, terminated, truncated = self.Env.Step(action)

		else:
			humanState = self.Env.Render()

			nextState, reward, terminated, truncated = self.Env.Step(action)

			if not self.HumanRender:
				humanState = None

			erStep = ERStep(self.Env._CurrentStep, humanState, self.State, reward, action, actionReason)
			self.EpisodeReplay.AddStep(erStep)




		truncated = truncated or self.StepCount >= self.MaxSteps
		self.StepCount += 1

		self.ExperienceStore.AddTransition(self.State, nextState, action, reward, terminated, truncated)

		self.TotalReward += reward

		self.State = nextState

		if terminated or truncated:

			if self.EpisodeReplay is not None:
				# save replay of the episode
				humanState = self.Env.Render()
				if not self.HumanRender:
					humanState = None

				erStep = ERStep(self.Env._CurrentStep, humanState, self.State, reward, None, None)
				self.EpisodeReplay.AddStep(erStep)

				self.EpisodeReplay.EpisodeEnd(terminated, truncated)
				self.EpisodeReplay.SaveToFolder(self.ReplayFolder)

				# save the episode to stats to table
				row = ConfigHelper.FlattenConfig(self.ReplayInfo, {
						"Terminated": terminated,
						"Truncated": truncated,
						"EpisodeTotalReward": self.TotalReward,
						"EpisodeSteps": self.StepCount
					})

				self._SaveToCsv(row)



			# log the episode to wandb
			self._Logger.EpisodeEnd(commit=False)
			self._Logger.LogDict({
				"Terminated": float(terminated),
				"Truncated": float(truncated),
				"EpisodeTotalReward": self.TotalReward,
				"EpisodeSteps": self.StepCount},
				commit=True)



		return nextState, reward, terminated, truncated


	def Reset(self) -> None:
		self.ExperienceStore.EmptyTransitionBuffer()
		self.State = self.Env.Reset()
		self.StepCount = 0
		self.TotalReward = 0

		if self.ReplayFolder is not None:
			self.EpisodeReplay = ER(self.ReplayInfo)
		return


	def _SaveToCsv(self, row):
		path = os.path.join(self.ReplayFolder, "stats.tsv")
		PathHelper.EnsurePathExists(path)

		if not os.path.exists(path):
			with open(path, "w") as f:
				f.write("\t".join(row.keys()) + "\n")

		with open(path, "a") as f:
			f.write("\t".join([str(x) for x in row.values()]) + "\n")

		return

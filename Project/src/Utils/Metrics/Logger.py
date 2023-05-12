import src.Utils.Singleton as Singleton
import wandb


class Logger(Singleton.Singleton):
	_ProjectName = "Dissertation"

	def Setup(self, config, runId=None, wandbOn=True):
		self._RunId = runId
		self._Config = config

		self._CurrentFrame = 0
		self._CurrentEpisode = 0
		self._TotalFrames = 0
		self._EpisodeCumulativeReward = 0


		self._WandbOn = wandbOn

		if self._WandbOn:
			wandb.init(project=self._ProjectName, config=self._Config , id=self._RunId, resume="allow")
		return

	def LogDict(self, dict):

		if self._WandbOn:
			wandb.log(dict, self._TotalFrames, commit=False)
		return


	def FrameEnd(self, frameReward, terminated, truncated):

		self._EpisodeCumulativeReward += frameReward

		logDict = {
			"TotalEpisodes": self._CurrentEpisode,
			"CurrentFrame": self._CurrentFrame,
			"FrameReward": frameReward,
			"EpisodeCumulativeReward": self._EpisodeCumulativeReward,
		}


		self._CurrentFrame += 1
		self._TotalFrames += 1

		if terminated or truncated:
			logDict["Terminated"] = float(terminated)
			logDict["Truncated"] = float(truncated)
			logDict["EpisodeTotalReward"] = self._EpisodeCumulativeReward

			self.EpisodeEnd()





		if self._WandbOn:
			self.LogDict(logDict)
			wandb.log({}, commit=True)
		return

	def EpisodeEnd(self):
		self._CurrentFrame = 0
		self._CurrentEpisode += 1
		self._EpisodeCumulativeReward = 0

		return
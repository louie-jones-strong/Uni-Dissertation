from typing import Optional
import typing
import Common.Utils.SharedCoreTypes as SCT
import Common.Utils.Singleton as Singleton
import wandb
import Common.Utils.Metrics.Timer as Timer

class Logger(Singleton.Singleton):
	# config
	_ProjectName = "Dissertation"
	_TimeStackSeparator = "."

	# state
	_Setup = False
	_TimerLabelStack:typing.List[str] = []

	def Setup(self, config:SCT.Config, runPath:str, runId:Optional[str] = None, wandbOn:bool = True) -> None:
		self._RunId = runId
		self._Config = config

		self._CurrentStep = 0
		self._CurrentEpisode = 0
		self._TotalSteps = 0
		self._EpisodeCumulativeReward = 0.0
		self._TotalTimePerStep:typing.Dict[str, float] = {}
		self._TotalTimePerEpisode:typing.Dict[str, float] = {}
		self._Setup = True




		self._WandbOn = wandbOn
		if self._WandbOn:
			wandb.init(project=self._ProjectName, config=self._Config, id=self._RunId, resume="allow", dir=runPath)
		return

	def LogDict(self, dict:typing.Dict[str, float]) -> None:
		if not self._Setup:
			return


		if self._WandbOn:
			wandb.log(dict, self._TotalSteps, commit=False)
		return


	def StepEnd(self, StepReward:SCT.Reward, terminated:bool, truncated:bool) -> None:
		if not self._Setup:
			return


		self._EpisodeCumulativeReward += StepReward

		logDict = {
			"TotalSteps": self._TotalSteps,
			"TotalEpisodes": self._CurrentEpisode,
			"CurrentStep": self._CurrentStep,
			"StepReward": StepReward,
			"EpisodeCumulativeReward": self._EpisodeCumulativeReward,
		}

		logDict.update(self._TotalTimePerStep)
		self._TotalTimePerStep.clear()




		if terminated or truncated:
			logDict["Terminated"] = float(terminated)
			logDict["Truncated"] = float(truncated)
			logDict["EpisodeTotalReward"] = self._EpisodeCumulativeReward

			logDict.update(self._TotalTimePerEpisode)
			self._TotalTimePerEpisode.clear()

			self._EpisodeEnd()

		if self._WandbOn:
			self.LogDict(logDict)
			wandb.log({}, commit=True)

		self._CurrentStep += 1
		self._TotalSteps += 1
		return

	def _EpisodeEnd(self) -> None:
		if not self._Setup:
			return

		self._CurrentStep = 0
		self._CurrentEpisode += 1
		self._EpisodeCumulativeReward = 0

		return


	def Time(self, label:str) -> Timer.Timer:

		assert self._Setup, "Logger not setup"
		assert len(label) > 0, "Timer label cannot be empty"
		assert self._TimeStackSeparator not in label, f"Timer label cannot contain '{self._TimeStackSeparator}'"

		self._TimerLabelStack.append(label)

		return Timer.Timer(label, self.TimerComplete)

	def TimerComplete(self, timer:Timer.Timer) -> None:
		assert self._Setup, "Logger not setup"

		label = timer._Label
		stackedLabel = self._TimeStackSeparator.join(self._TimerLabelStack)
		assert label == self._TimerLabelStack[-1], f"Timer '{stackedLabel}' completed out of order"

		self._TimerLabelStack.pop()


		StepLabel = f"Time:Step.{stackedLabel}"
		if StepLabel not in self._TotalTimePerStep:
			self._TotalTimePerStep[StepLabel] = 0.0
		self._TotalTimePerStep[StepLabel] += timer._Interval

		episodeLabel = f"Time:Episode.{stackedLabel}"
		if episodeLabel not in self._TotalTimePerEpisode:
			self._TotalTimePerEpisode[episodeLabel] = 0.0
		self._TotalTimePerEpisode[episodeLabel] += timer._Interval

		return
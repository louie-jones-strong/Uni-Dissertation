from typing import Optional
import typing
import src.Common.Utils.SharedCoreTypes as SCT
import src.Common.Utils.Singleton as Singleton
import wandb
import src.Common.Utils.Metrics.Timer as Timer
from wandb.keras import WandbCallback

class Logger(Singleton.Singleton):
	# config
	_ProjectName = "Dissertation-v2"
	_TimeStackSeparator = "."

	# state
	_Setup = False
	_TimerLabelStack:typing.List[str] = []

	def Setup(self, config:SCT.Config, runPath:str, runId:Optional[str] = None, wandbOn:bool = True) -> None:
		self._RunId = runId
		self._Config = config

		self._TotalTimePerStep:typing.Dict[str, float] = {}
		self._TotalTimePerEpisode:typing.Dict[str, float] = {}
		self._Setup = True




		self._WandbOn = wandbOn
		if self._WandbOn:
			wandb.init(project=self._ProjectName, config=self._Config, id=self._RunId, resume="allow", dir=runPath)
		return

	def LogDict(self, dict:typing.Dict[str, float], commit:bool = True) -> None:
		if not self._Setup:
			return


		if self._WandbOn:
			wandb.log(dict, commit=commit)
		return


	def GetFitCallback(self):
		if self._WandbOn:
			return WandbCallback()

		return None

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
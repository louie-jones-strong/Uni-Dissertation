from src.Common.EpisodeReplay.EpisodeReplayStep import EpisodeReplayStep as ERStep
import src.Common.Utils.Formatter as Formatter
import time
import os
import pickle
import typing
from typing import Optional, Any


class EpisodeReplay:

	def __init__(self, replayInfo:Optional[typing.Dict[str, Any]] = None) -> None:

		self.ReplayInfo = replayInfo

		self.Steps:typing.List[ERStep] = []
		self.Terminated = False
		self.Truncated = False


		self.StartTime = time.time_ns()
		self.EndTime:Optional[int] = None


		# create a unique id for the episode
		# self.EpisodeId = str(uuid.uuid4())
		# self.EpisodeId = str(self.StartTime)
		self.EpisodeId = Formatter.ConvertNsTime(self.StartTime)
		return

	def AddStep(self, step:ERStep) -> None:
		self.Steps.append(step)
		return

	def EpisodeEnd(self, terminated:bool, truncated:bool) -> None:
		self.Terminated = terminated
		self.Truncated = truncated
		self.EndTime = time.time_ns()
		return


# region formatting
	def DurationText(self, endTime:Optional[int] = None) -> str:

		if endTime is None:
			endTime = self.EndTime

		return Formatter.ConvertNsDuration(endTime - self.StartTime)

	def ReasonEnded(self) -> str:
		if self.Terminated:
			return "Terminated"
		elif self.Truncated:
			return "Truncated"
		else:
			return "Unknown"

	def NumSteps(self) -> int:
		return len(self.Steps)

	def TotalReward(self) -> float:
		return sum([step.Reward for step in self.Steps])
# endregion formatting



# region File IO
	def SaveToFolder(self, folderPath:str) -> str:

		# define folder to store all the data for this episode
		replayFolder = os.path.join(folderPath, self.EpisodeId)

		# create the run path
		if not os.path.exists(replayFolder):
			os.makedirs(replayFolder)

		filePath = os.path.join(replayFolder, "data.pkl")

		with open(filePath, 'wb') as f:
			selfDict = {
				"Terminated": self.Terminated,
				"Truncated": self.Truncated,
				"EpisodeId": self.EpisodeId,
				"StartTime": self.StartTime,
				"EndTime": self.EndTime,
				"ReplayInfo": self.ReplayInfo
			}
			stepsData = []

			for step in self.Steps:
				stepDict = step.Save(replayFolder)
				stepsData.append(stepDict)

			selfDict["Steps"] = stepsData

			pickle.dump(selfDict, f)
			# f.write(jsonStr)
		return replayFolder




	@classmethod
	def LoadFromFolder(cls, folderPath:str) -> 'EpisodeReplay':

		filePath = os.path.join(folderPath, "data.pkl")

		with open(filePath, 'rb') as f:
			data = pickle.load(f)

		instance = cls()

		instance.Terminated = data.get("Terminated", False)
		instance.Truncated = data.get("Truncated", False)
		instance.EpisodeId = data.get("EpisodeId", None)
		instance.StartTime = data.get("StartTime", None)
		instance.EndTime = data.get("EndTime", None)

		instance.Steps = []
		stepsData = data.get("Steps", [])
		for stepData in stepsData:
			step = ERStep.Load(stepData, folderPath)
			instance.Steps.append(step)

		return instance
# endregion File IO
from src.Common.EpisodeReplay.EpisodeReplayStep import EpisodeReplayStep as ERStep
import src.Common.Utils.Formatter as Formatter
import time
import uuid
import os
import pickle

class EpisodeReplay:

	def __init__(self, replayInfo=None) -> None:

		self.ReplayInfo = replayInfo

		self.Steps = []
		self.Terminated = False
		self.Truncated = False


		self.StartTime = time.time_ns()
		self.EndTime = None


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


#region formatting
	def DurationText(self, endTime=None) -> str:

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
#endregion formatting



#region File IO
	def SaveToFolder(self, folderPath):

		# define folder to store all the data for this episode
		replayFolder = os.path.join(folderPath, self.EpisodeId)

		# create the run path
		if not os.path.exists(replayFolder):
			os.makedirs(replayFolder)

		filePath = os.path.join(replayFolder, "data.pkl")

		with open(filePath, 'wb') as f:
			selfDict = {
				"Steps": [],
				"Terminated": self.Terminated,
				"Truncated": self.Truncated,
				"EpisodeId": self.EpisodeId,
				"StartTime": self.StartTime,
				"EndTime": self.EndTime,
				"ReplayInfo": self.ReplayInfo
			}

			for step in self.Steps:
				stepDict = step.Save(replayFolder)
				selfDict["Steps"].append(stepDict)

			pickle.dump(selfDict, f)
			# f.write(jsonStr)
		return




	@classmethod
	def LoadFromFolder(cls, folderPath):

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
#endregion File IO
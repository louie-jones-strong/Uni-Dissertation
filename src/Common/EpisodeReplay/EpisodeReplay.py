from src.Common.EpisodeReplay.EpisodeReplayStep import EpisodeReplayStep as ERStep
import src.Common.Utils.Formatter as Formatter
import json
import time
import uuid
import os

class EpisodeReplay:

	def __init__(self):
		self.Steps = []
		self.Terminated = False
		self.Truncated = False

		# create a unique id for the episode
		self.EpisodeId = str(uuid.uuid4())

		self.StartTime = time.time_ns()
		self.EndTime = None
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
	def DurationText(self) -> str:
		return Formatter.ConvertNs(self.EndTime - self.StartTime)

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


#region json serialization
	def ToJson(self):

		selfDict = {
			"Steps": [],
			"Terminated": self.Terminated,
			"Truncated": self.Truncated,
			"EpisodeId": self.EpisodeId,
			"StartTime": self.StartTime,
			"EndTime": self.EndTime
		}

		for step in self.Steps:
			selfDict["Steps"].append(step.__dict__())


		return json.dumps(selfDict, indent=4)

	@classmethod
	def FromJson(cls, jsonStr):
		data = json.loads(jsonStr)

		instance = cls()

		instance.Terminated = data.get("Terminated", False)
		instance.Truncated = data.get("Truncated", False)
		instance.EpisodeId = data.get("EpisodeId", None)
		instance.StartTime = data.get("StartTime", None)
		instance.EndTime = data.get("EndTime", None)

		stepsData = data.get("Steps", [])
		steps = [ERStep.FromJson(stepData) for stepData in stepsData]
		instance.Steps = steps

		return instance
#endregion json serialization


#region File IO
	def SaveToFolder(self, folderPath):

		# create the run path
		if not os.path.exists(folderPath):
			os.makedirs(folderPath)

		filePath = os.path.join(folderPath, f"{self.EpisodeId}.json")

		with open(filePath, 'w') as f:
			f.write(self.ToJson())
		return

	@classmethod
	def LoadFromFile(cls, filePath):
		with open(filePath, 'r') as f:
			jsonStr = f.read()
		return cls.FromJson(jsonStr)
#endregion File IO
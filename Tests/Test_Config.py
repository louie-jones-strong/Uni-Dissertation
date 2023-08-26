import unittest
import src.Common.Utils.Config.ConfigHelper as ConfigHelper
import os

class Test_Config(unittest.TestCase):
	ConfigRoot = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Config")
	EnvsRoot = os.path.join(ConfigRoot, "Envs")

	def test_Folders(self):

		# check folders
		self.assertTrue(os.path.exists(self.ConfigRoot), "ConfigRoot does not exist")
		self.assertTrue(os.path.exists(self.EnvsRoot), "EnvsRoot does not exist")

		files = os.listdir(self.EnvsRoot)
		self.assertTrue(len(files) > 0, "No files in EnvsRoot")

		for file in files:
			filePath = os.path.join(self.EnvsRoot, file)
			self.assertTrue(os.path.isfile(filePath), "File in EnvsRoot is not a file")
			self.assertTrue(os.path.exists(filePath), "File in EnvsRoot does not exist")
			self.assertTrue(file.endswith(".json"), "File in EnvsRoot does not end with .json")

			config = ConfigHelper.LoadConfig(filePath)

			self.CheckKey(config, "Name", str)
			self.CheckKey(config, "EnvType", str, allowedValues=["Gym"])

			if config["EnvType"] == "Gym":
				self.CheckKey(config, "GymConfig", dict)


			self.CheckSpaceConfig(config, "ObservationSpace")
			self.CheckSpaceConfig(config, "ActionSpace")


			self.CheckKey(config, "StepRewardRange", list)
			self.CheckKey(config, "EpisodeRewardRange", list)

			self.CheckKey(config, "IsDeterministic", bool)
			self.CheckKey(config, "ClipRewards", bool)


			self.CheckKey(config, "MaxSteps", int)
			self.CheckKey(config, "MaxEpisodes", int)
			self.CheckKey(config, "NumEnvsPerWorker", int)

			self.CheckAgentConfig(config)
		return






	def CheckKey(self, config, key, type, allowedValues=None):

		self.assertTrue(key in config, f"Key: {key} not found in config")

		value = config[key]
		self.assertTrue(isinstance(value, type), f"Value is not of type: {type}")

		if type == str:
			self.assertTrue(len(value) > 0, "Value is empty string")

			if allowedValues is not None:
				self.assertTrue(value in allowedValues, f"Value is not in allowed values: {allowedValues}")
		return

	def CheckSpaceConfig(self, config, key):

		self.CheckKey(config, key, dict)
		spaceConfig = config[key]

		self.CheckKey(spaceConfig, "Type", str, allowedValues=["Box", "Discrete"])

		spaceType = spaceConfig["Type"]

		if spaceType == "Box":
			self.CheckKey(spaceConfig, "Shape", list)

		elif spaceType == "Discrete":
			self.CheckKey(spaceConfig, "Shape", int)


		return

	def CheckAgentConfig(self, config):

		self.CheckKey(config, "AgentConfig", dict)
		# agentConfig = config["AgentConfig"]

		return



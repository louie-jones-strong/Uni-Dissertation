import unittest
import os
import src.Worker.Worker as Worker
import src.Common.Store.ExperienceStore.EsBase as EsBase
import src.Common.Utils.ConfigHelper as ConfigHelper
from src.Common.Enums.eAgentType import eAgentType
import src.Common.Utils.ModelHelper as ModelHelper
import src.Common.Store.ModelStore.MsBase as MsBase

class Test_Agents(unittest.TestCase):

	ConfigRoot = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Config')
	EnvsRoot = os.path.join(ConfigRoot, 'Envs')

	MaxEpisodesOverride = 2
	MaxStepsOverride = 50
	MaxWorkersOverride = 1

	def test_Random(self):
		self.CheckAgent(eAgentType.Random)
		return

	# def test_HardCoded(self):
	# 	self.CheckAgent(eAgentType.HardCoded)
	# 	return

	def test_ML(self):
		self.CheckAgent(eAgentType.ML)
		return




	def CheckAgent(self, agentType):

		configList = os.listdir(self.EnvsRoot)

		for config in configList:
			configPath = os.path.join(self.EnvsRoot, config)
			# check is a file
			self.assertTrue(os.path.isfile(configPath), 'Config is not a file')

			# check is a json file
			self.assertTrue(config.endswith('.json'), 'Config is not a json file')

			envConfig = ConfigHelper.LoadConfig(configPath)

			# modify the config to use a smaller number of episodes and steps
			envConfig["MaxEpisodes"] = self.MaxEpisodesOverride
			envConfig["MaxSteps"] = self.MaxStepsOverride
			envConfig["NumEnvsPerWorker"] = self.MaxWorkersOverride


			modelStore = MsBase.MsBase()

			modelHelper = ModelHelper.ModelHelper()
			modelHelper.Setup(envConfig, modelStore)




			experienceStore = EsBase.EsBase()
			worker = Worker.Worker(envConfig, agentType, True, experienceStore)

			worker.Run()
		return

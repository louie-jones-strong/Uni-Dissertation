import unittest
import os
import src.Worker.Worker as Worker
import src.Common.Store.ExperienceStore.EsBase as EsBase
import src.Common.Utils.ConfigHelper as ConfigHelper
from src.Common.Enums.eAgentType import eAgentType
import src.Common.Utils.PathHelper as PathHelper


class Test_Envs(unittest.TestCase):

	ConfigRoot = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Config')
	EnvsRoot = os.path.join(ConfigRoot, 'Envs')
	MaxEpisodesOverride = 1
	MaxStepsOverride = 10
	AgentType = eAgentType.Random

	def test_Envs(self):
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
			envConfig["NumEnvsPerWorker"] = 2

			experienceStore = EsBase.EsBase()
			worker = Worker.Worker(envConfig, self.AgentType, True, experienceStore)

			worker.Run()



		return

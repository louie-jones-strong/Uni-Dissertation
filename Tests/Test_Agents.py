import unittest
import os
import src.Worker.Worker as Worker
import src.Common.Store.ExperienceStore.EsBase as EsBase
from src.Common.Enums.eAgentType import eAgentType
import src.Common.Utils.ModelHelper as ModelHelper
import src.Common.Store.ModelStore.MsBase as MsBase
import src.Worker.EnvRunner as EnvRunner
import src.Worker.Environments.BaseEnv as BaseEnv
import src.Common.Utils.Config.ConfigManager as ConfigManager

class Test_Agents(unittest.TestCase):

	ConfigRoot = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Config')
	EnvsRoot = os.path.join(ConfigRoot, 'Envs')

	MaxEpisodesOverride = 2
	MaxStepsOverride = 20
	MaxWorkersOverride = 1

	def test_Random(self):
		self.CheckAgent(eAgentType.Random)
		return

	def test_HardCoded(self):
		self.CheckAgent(eAgentType.HardCoded)
		return

	def test_ML(self):
		self.CheckAgent(eAgentType.ML)
		return




	def CheckAgent(self, agentType):

		configManager = ConfigManager.ConfigManager()


		configList = os.listdir(self.EnvsRoot)

		for config in configList:
			configManager.Setup(config)
			configManager.Config["UseRealSim"] = True
			configManager.Config["MonteCarloConfig"]["MaxSecondsPerAction"] = 0.01
			configManager.Config["MonteCarloConfig"]["RollOutConfig"]["MaxRollOutCount"] = 2
			configManager.Config["MonteCarloConfig"]["RollOutConfig"]["MaxRollOutDepth"] = 1

			configPath = os.path.join(self.EnvsRoot, config)
			# check is a file
			self.assertTrue(os.path.isfile(configPath), 'Config is not a file')

			# check is a json file
			self.assertTrue(config.endswith('.json'), 'Config is not a json file')

			# modify the config to use a smaller number of episodes and steps
			configManager.EnvConfig["MaxEpisodes"] = self.MaxEpisodesOverride
			configManager.EnvConfig["MaxSteps"] = self.MaxStepsOverride
			configManager.EnvConfig["NumEnvsPerWorker"] = self.MaxWorkersOverride


			modelStore = MsBase.MsBase()

			modelHelper = ModelHelper.ModelHelper()
			modelHelper.Setup(configManager.EnvConfig, modelStore)


			envRunners = []

			env = BaseEnv.GetEnv(configManager.EnvConfig, render=False)
			experienceStore = EsBase.EsBase()
			runner = EnvRunner.EnvRunner(env, configManager.EnvConfig["MaxSteps"], experienceStore)
			envRunners.append(runner)

			worker = Worker.Worker(agentType, envRunners, True)

			worker.Run()
		return

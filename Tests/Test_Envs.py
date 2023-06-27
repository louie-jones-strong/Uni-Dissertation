import unittest
import os
import main

class Test_Envs(unittest.TestCase):
	ConfigRoot = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Config')
	EnvsRoot = os.path.join(ConfigRoot, 'Envs')
	MaxEpisodesOverride = 1
	MaxStepsOverride = 10
	AgentType = 'Random'

	def test_Envs(self):
		configList = os.listdir(self.EnvsRoot)

		for config in configList:
			configPath = os.path.join(self.EnvsRoot, config)
			# check is a file
			self.assertTrue(os.path.isfile(configPath), 'Config is not a file')

			# check is a json file
			self.assertTrue(config.endswith('.json'), 'Config is not a json file')


			# Main.Main(
			# 	envConfigPath=configPath,
			# 	isPlayMode=False,
			# 	load=False,
			# 	agentType=self.AgentType,
			# 	wandbOn=False,
			# 	profileOn=False,
			# 	maxEpisodesOverride=self.MaxEpisodesOverride, maxStepsOverride=self.MaxStepsOverride)



		return

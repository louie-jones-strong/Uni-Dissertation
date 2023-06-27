import unittest
import os

class Test_Config(unittest.TestCase):
	ConfigRoot = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Config')
	EnvsRoot = os.path.join(ConfigRoot, 'Envs')

	def test_Folders(self):

		self.assertTrue(os.path.exists(self.ConfigRoot), 'ConfigRoot does not exist')

		self.assertTrue(os.path.exists(self.EnvsRoot), 'EnvsRoot does not exist')

		return

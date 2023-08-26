import unittest
import src.Common.Utils.Config.ConfigHelper as ConfigHelper

class Test_ConfigHelper(unittest.TestCase):

	def test_FlattenConfig(self):
		self.assertIsNotNone(ConfigHelper.FlattenConfig)

		config = {
			"test1": {
				"test2": {
					"test3": {
						"test4": 1,
						"test5": 2
					}
				}
			}
		}

		flatConfig = ConfigHelper.FlattenConfig(config)

		self.assertEqual(list(flatConfig.keys()), ["test1_test2_test3_test4", "test1_test2_test3_test5"])

		return




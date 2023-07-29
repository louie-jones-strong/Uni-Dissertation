# import unittest
# import src.Common.Store.ExperienceStore.EsNumpy as EsNumpy
# import src.Common.Utils.PathHelper as PathHelper
# import os


# todo add back but needs to create run data for it to work on github actions
# class Test_EsNumpy(unittest.TestCase):

# 	def test_Numpy(self):
# 		self.assertNotEqual(EsNumpy, None)

# 		runPath = os.path.join(PathHelper.GetRootPath(), "Data","FrozenLake", str(1))

# 		instance = EsNumpy.EsNumpy(runPath)
# 		self.assertNotEqual(instance, None)
# 		self.assertNotEqual(instance.Load, None)
# 		self.assertNotEqual(instance.Save, None)

# 		# check data is empty
# 		self.assertEqual(instance.States, None)

# 		# load data
# 		instance.Load()


# 		self.assertFalse(instance.States is None)



# 		return
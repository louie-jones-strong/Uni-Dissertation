import unittest
import src.Common.Utils.Metrics.Timer as Timer

class Test_Timer(unittest.TestCase):

	def test_CanInstantiate(self):
		timer = Timer.Timer("Test")
		self.assertIsNotNone(timer)
		return

	def test_CanUseAsContextManager(self):
		timer = Timer.Timer("Test")
		with timer:
			pass
		return

	def test_CollectsTime(self):
		timer = Timer.Timer("Test")
		with timer:
			for i in range(10_000_000):
				_ = i * i

		self.assertGreater(timer._Interval, 0.0)
		return

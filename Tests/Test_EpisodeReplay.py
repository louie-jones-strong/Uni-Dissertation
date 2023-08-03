import unittest

import src.Common.EpisodeReplay.EpisodeReplay as ER
import src.Common.EpisodeReplay.EpisodeReplayStep as ERStep
import time


class Test_EpisodeReplay(unittest.TestCase):

	def test_ClassExists(self):
		self.assertIsNotNone(ER.EpisodeReplay)
		self.assertIsNotNone(ERStep.EpisodeReplayStep)
		return

	def test_CanInstantiate(self):
		er = ER.EpisodeReplay()
		self.assertIsNotNone(er)

		erStep = ERStep.EpisodeReplayStep(None, None, None, None, None, None)
		self.assertIsNotNone(erStep)

		# check methods
		self.assertIsNotNone(er.AddStep)
		self.assertIsNotNone(er.EpisodeEnd)
		self.assertIsNotNone(er.ToJson)
		self.assertIsNotNone(er.FromJson)
		return

	def test_Timing(self):
		nsMultiplier = 1_000_000_000
		perStepTime = 0.01
		numSteps = 3
		allowedDelta = 0.05


		er = ER.EpisodeReplay()

		# check start time
		startTime = er.StartTime
		self.assertGreater(startTime, 0)
		self.assertIsNone(er.EndTime)
		self.assertFalse(er.Terminated)
		self.assertFalse(er.Truncated)


		# add some steps
		for i in range(numSteps):
			time.sleep(perStepTime)
			step = ERStep.EpisodeReplayStep(None, None, None, None, None, None)
			er.AddStep(step)

			expectedTime = startTime + (i+1) * perStepTime * nsMultiplier
			self.assertAlmostEqual(step.CompletedTime, expectedTime, delta=allowedDelta * nsMultiplier)

		# end the episode
		er.EpisodeEnd(terminated=True, truncated=False)

		# check end time
		self.assertAlmostEqual(startTime, er.StartTime, delta=allowedDelta * nsMultiplier)
		expectedTime = startTime + (i+1) * perStepTime * nsMultiplier
		self.assertGreaterEqual(er.EndTime, expectedTime)

		self.assertTrue(er.Terminated)
		self.assertFalse(er.Truncated)
		return

	def test_JsonConversion(self):
		er = ER.EpisodeReplay()
		er.AddStep(ERStep.EpisodeReplayStep(None, None, None, None, None, None))
		er.AddStep(ERStep.EpisodeReplayStep(None, None, None, None, None, None))
		er.EpisodeEnd(terminated=True, truncated=False)


		jsonStr = er.ToJson()
		self.assertIsNotNone(jsonStr)

		er2 = ER.EpisodeReplay.FromJson(jsonStr)
		self.assertIsNotNone(er2)

		self.assertEqual(er.Terminated, er2.Terminated)
		self.assertEqual(er.Truncated, er2.Truncated)
		self.assertEqual(er.EpisodeId, er2.EpisodeId)
		self.assertEqual(er.StartTime, er2.StartTime)
		self.assertEqual(er.EndTime, er2.EndTime)
		self.assertEqual(len(er.Steps), len(er2.Steps))


		return
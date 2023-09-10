import unittest
from src.Worker.Agents.MonteCarloAgent import MonteCarloAgent
from gymnasium.spaces import Discrete

class Test_MCTS(unittest.TestCase):

	def test_SampleActions_Same(self):

		actionSpace = Discrete(2)
		samples = MonteCarloAgent._SampleActions(actionSpace, 2)

		self.assertEqual(len(samples), 2, "Number of samples is not correct")
		self.assertCountEqual(samples, [0, 1], "Samples are not correct")


		return
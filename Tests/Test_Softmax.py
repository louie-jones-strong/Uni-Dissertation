import unittest
from src.Worker.Agents.BaseAgent import BaseAgent
import numpy as np

class Test_Softmax(unittest.TestCase):




	def test_Same(self):

		values = np.array([1, 1])
		output = BaseAgent._SoftMax(values, temperature=1)


		self.assertEqual(output[0], 0.5)
		self.assertEqual(output[1], 0.5)

		return

	def test_Negative(self):

		values = np.array([-1, 1])
		output = BaseAgent._SoftMax(values, temperature=1)


		self.assertEqual(output[0], 0)
		self.assertEqual(output[1], 1)

		return

	def test_Zero(self):

		values = np.array([0, 1])
		output = BaseAgent._SoftMax(values, temperature=1)


		self.assertEqual(output[0], 0)
		self.assertEqual(output[1], 1)

		return




	def test_Temp_Zero(self):

		values = np.array([1, 2])
		output = BaseAgent._SoftMax(values, temperature=0)


		self.assertEqual(output[0], 0)
		self.assertEqual(output[1], 1)

		return
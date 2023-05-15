import unittest
import os
from src.Agents.Predictors import DecisonTreePredictor, LinearRegressionPredictor, EnsemblePredictor
import src.DataManager.DataColumnTypes as DCT
import src.DataManager.DataManager as DataManager
import gymnasium.spaces as spaces
from src.Utils.PathHelper import GetRootPath

class Test_Predictors(unittest.TestCase):

	def setUp(self):
		self.PredictorConstructors = [
			DecisonTreePredictor.DecisonTreePredictor,
			LinearRegressionPredictor.LinearRegressionPredictor,
			EnsemblePredictor.EnsemblePredictor
		]

		self.ObservationSpace = spaces.Discrete(16)
		self.ActionSpace = spaces.Discrete(4)
		self.DataPath = os.path.join(GetRootPath(), "data", "FrozenLake")

		self.DataManager = DataManager.DataManager()
		self.DataManager.Setup({}, self.ObservationSpace, self.ActionSpace)
		return

	def test_ClassesExist(self):

		for predictor in self.PredictorConstructors:
			self.assertIsNotNone(predictor, f"{predictor} is None")

		return

	def test_CanInstantiate(self):
		xLabels = [DCT.DataColumnTypes.CurrentState]
		yLabels = [DCT.DataColumnTypes.Terminated]

		x = [[1]]
		y = [[False]]
		instances = []

		for predictor in self.PredictorConstructors:
			predictorInstance = predictor(xLabels, yLabels)
			self.assertIsNotNone(predictorInstance, f"{predictor} is None")

			# check that the predictor has the correct name
			expectedName = f"{yLabels[0].name}_{predictor.__name__.replace('Predictor', '')}"
			self.assertEqual(predictorInstance._Name, expectedName)


			# check predict returns None when not trained
			prediction, confidence = predictorInstance.Predict(x)

			self.assertIsNone(prediction)

			# check no train when no data
			predictorInstance.Observe(x, y)
			self.assertEqual(predictorInstance._FramesSinceTrained, -1)

			predictorInstance.Train()
			self.assertEqual(predictorInstance._FramesSinceTrained, -1)

			instances.append(predictorInstance)


		# load data into the data manager
		self.assertTrue(os.path.exists(self.DataPath), f"Data path {self.DataPath} does not exist")
		self.DataManager.Load(self.DataPath)
		x, y = self.DataManager.GetXYData(xLabels, yLabels)
		self.assertGreater(len(x[0]), 1)

		# check
		for instance in instances:
			instance.Train()
			prediction, confidence = instance.Predict(x)
			self.assertIsNotNone(prediction)
			self.assertIsNotNone(confidence)
			self.assertGreaterEqual(confidence, 0)
			self.assertLessEqual(confidence, 1)
		return


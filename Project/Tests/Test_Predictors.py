import unittest
import os
from src.Agents.Predictors import DecisonTreePredictor, LinearPredictor, EnsemblePredictor, MultiYPredictor
import src.DataManager.DataColumnTypes as DCT
import src.DataManager.DataManager as DataManager
import gymnasium.spaces as spaces
from src.Utils.PathHelper import GetRootPath

class Test_Predictors(unittest.TestCase):

	def setUp(self):
		self.PredictorConstructors = [
			DecisonTreePredictor.DecisonTreePredictor,
			LinearPredictor.LinearPredictor,
			EnsemblePredictor.EnsemblePredictor,
		]

		self.ObservationSpace = spaces.Discrete(16)
		self.ActionSpace = spaces.Discrete(4)
		self.RewardRange = (0, 1)
		self.DataPath = os.path.join(GetRootPath(), "data", "FrozenLake")

		self.DataManager = DataManager.DataManager()
		self.DataManager.Setup({}, self.ObservationSpace, self.ActionSpace, self.RewardRange)
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
			self.CheckPredictorName(predictorInstance, yLabels)


			# check predict returns None when not trained
			prediction, confidence = predictorInstance.Predict(x)

			self.assertIsNone(prediction)

			# check no train when no data
			predictorInstance.Observe(x, y)
			self.assertEqual(predictorInstance._StepsSinceTrained, -1)

			predictorInstance.Train()
			self.assertEqual(predictorInstance._StepsSinceTrained, -1)

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


	def test_MultiYPredictor(self):
		xLabels = [DCT.DataColumnTypes.CurrentState]
		yLabels = [DCT.DataColumnTypes.Terminated, DCT.DataColumnTypes.Truncated]

		x = [[1]]
		y = [[False], [False]]

		predictorInstance = MultiYPredictor.MultiYPredictor(xLabels, yLabels)
		self.assertIsNotNone(predictorInstance, f"{predictorInstance} is None")

		self.CheckPredictorName(predictorInstance, yLabels)

		# check predict returns None when not trained
		prediction, confidence = predictorInstance.Predict(x)

		self.assertIsNone(prediction)

		# check no train when no data
		predictorInstance.Observe(x, y)
		self.assertEqual(predictorInstance._StepsSinceTrained, -1)

		predictorInstance.Train()
		self.assertEqual(predictorInstance._StepsSinceTrained, -1)


		# load data into the data manager
		self.assertTrue(os.path.exists(self.DataPath), f"Data path {self.DataPath} does not exist")
		self.DataManager.Load(self.DataPath)
		x, y = self.DataManager.GetXYData(xLabels, yLabels)
		self.assertGreater(len(x[0]), 1)


		predictorInstance.Train()
		prediction, confidence = predictorInstance.Predict(x)
		self.assertIsNotNone(prediction)
		self.assertIsNotNone(confidence)
		self.assertGreaterEqual(confidence, 0)
		self.assertLessEqual(confidence, 1)


		return


	def CheckPredictorName(self, predictor, yLabels):
		yLabelNames = "".join([y.name for y in yLabels])
		yLabelNames = yLabelNames.replace("DataColumnTypes.", "")
		expectedName = f"{yLabelNames}_{predictor.__class__.__name__.replace('Predictor', '')}"
		self.assertEqual(predictor._Name, expectedName)
		return



	def test_PredictingForwardModel(self):
		xLabels = [DCT.DataColumnTypes.CurrentState, DCT.DataColumnTypes.Action]
		yLabels = [
			DCT.DataColumnTypes.NextState,
			DCT.DataColumnTypes.Terminated,
			DCT.DataColumnTypes.Truncated,
			DCT.DataColumnTypes.Reward]

		predictor = MultiYPredictor.MultiYPredictor(xLabels, yLabels)

		self.assertTrue(os.path.exists(self.DataPath), f"Data path {self.DataPath} does not exist")
		self.DataManager.Load(self.DataPath)



		predictor.Train()
		self.assertGreaterEqual(predictor._StepsSinceTrained, 0)

		x = [[1], [0]]
		y = [[1], [False], [False], [0.0]]

		predictor.Predict(x)
		predictor.Observe(x, y)

		return
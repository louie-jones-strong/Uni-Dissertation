import unittest
import src.Agents.Predictors.DecisonTreePredictor as DecisionTreePredictor

class Test_Predictors(unittest.TestCase):

	def test_DecisonTreePredictor(self):

		self.assertIsNotNone(DecisionTreePredictor,
			"Predictor file does not exist")

		self.assertIsNotNone(DecisionTreePredictor.DecisonTreePredictor,
			"class DecisonTreePredictor does not exist in Predictor file")

		predictor = DecisionTreePredictor.DecisonTreePredictor("x", "y")

		self.assertIsNotNone(predictor,
			"DecisonTreePredictor could not be instantiated")

		# check methods
		self.assertIsNotNone(predictor.PredictValue,
			"DecisonTreePredictor does not have a Predict method")

		self.assertIsNotNone(predictor.Train,
			"DecisonTreePredictor does not have a Predict method")

		# check prediction
		predictor.Train()


		x = [1, 2, 3]
		predictions, confidence = predictor.PredictValue(x)




		return

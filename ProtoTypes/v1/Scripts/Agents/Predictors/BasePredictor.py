
class BasePredictor:

	def __init__(self, dataManager, xLabel, yLabel):
		self.DataManager = dataManager
		self.XLabel = xLabel
		self.YLabel = yLabel

		return

	def LoadConfig(self, config):
		return

	def Save(self, path):
		return

	def Load(self, path):
		return

	def Train(self):
		return

	def PredictValue(self, x):
		y = None
		confidence = None
		return y, confidence

	def GetValueNovelties(self, x):
		novelties = None
		return novelties


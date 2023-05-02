import DataManager.DataManager as DataManager

class BasePredictor:

	def __init__(self, xLabel, yLabel):
		self.DataManager = DataManager.DataManager()
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

	def _GetSamples(self):
		indexs, priorities, columns = self.DataManager.GetColumns([self.XLabel, self.YLabel])

		return


	def PredictValue(self, x):
		y = None
		confidence = None
		return y, confidence

	def GetValueNovelties(self, x):
		novelties = None
		return novelties


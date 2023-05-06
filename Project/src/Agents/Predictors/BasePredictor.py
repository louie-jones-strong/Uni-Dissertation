import src.DataManager.DataManager as DataManager

class BasePredictor:

	def __init__(self, xLabel, yLabel):
		self.DataManager = DataManager.DataManager()
		self.XLabel = xLabel
		self.YLabel = yLabel

		return

	def LoadConfig(self, config):
		return

	def Save(self, folderPath:str):
		return

	def Load(self, folderPath:str):
		return

	def Train(self):
		return

	def _GetSamples(self):
		indexs, priorities, columns = self.DataManager.GetColumns([self.XLabel, self.YLabel])

		x = columns[:len(self.XLabel)]
		y = columns[len(self.XLabel):]

		return x, y


	def PredictValue(self, x):
		y = None
		confidence = None
		return y, confidence

	def GetValueNovelties(self, x):
		novelties = None
		return novelties


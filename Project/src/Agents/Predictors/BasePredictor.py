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
		joinedColumns = self.XLabel + self.YLabel
		indexs, priorities, columns = self.DataManager.Sample(joinedColumns)

		x = columns[:len(self.XLabel)]
		y = columns[len(self.XLabel):]

		return x, y


	def PredictValue(self, x):
		y = None
		return y

	def GetValueNovelties(self, x):
		novelties = None
		return novelties


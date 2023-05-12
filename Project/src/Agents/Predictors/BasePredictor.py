import src.DataManager.DataManager as DataManager
import src.Utils.Metrics.Logger as Logger

class BasePredictor:

	def __init__(self, xLabels, yLabels):
		self._DataManager = DataManager.DataManager()
		self._Logger = Logger.Logger()

		self._XLabels = xLabels
		self._YLabels = yLabels

		self._FramesSinceTrained = -1
		return

	def LoadConfig(self, config):
		return

	def Save(self, folderPath:str):
		return

	def Load(self, folderPath:str):
		return








	def Predict(self, x):
		predicted = None
		return predicted



	def Observe(self, x, y):
		self._FramesSinceTrained += 1

		predicted = self.Predict(x)



		# do we need to train this frame?
		self.Train()
		return

	def Train(self):
		self._FramesSinceTrained = 0
		return

	def _Evaluate(self, validationData=None):
		if validationData is None:
			validationData = self._DataManager.GetXYData(self._XLabels, self._YLabels)

		x, y = validationData

		# create a confusion matrix if the problem is discrete



		return
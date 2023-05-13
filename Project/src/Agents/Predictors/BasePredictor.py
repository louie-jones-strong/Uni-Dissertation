import src.DataManager.DataManager as DataManager
import src.Utils.Metrics.Logger as Logger
import src.DataManager.DataColumnTypes as DCT
import typing


class BasePredictor:

	def __init__(self,
			xLabels:typing.List[DCT.DataColumnTypes],
			yLabels:typing.List[DCT.DataColumnTypes],
			name:str):

		self._XLabels = xLabels
		self._YLabels = yLabels
		self._Name = name


		self._DataManager = DataManager.DataManager()
		self._Logger = Logger.Logger()

		self._FramesSinceTrained = -1

		return

	def LoadConfig(self, config) -> None:
		return

	def Save(self, folderPath:str) -> None:
		return

	def Load(self, folderPath:str) -> None:
		return





	def Predict(self, x):

		proccessedX = self._PreProccessX(x)
		predicted = self._Predict(proccessedX)
		proccessedPrediction = self._PostProccess(predicted)

		confidence = 0.5
		return proccessedPrediction, confidence

	def Observe(self, x, y):

		proccessedY = self._PreProccessY(y)
		predicted = self.Predict(x)[0]

		error = abs(proccessedY - predicted)
		self._Logger.LogDict({
			f"{self._Name}_Error": error
		})

		self._FramesSinceTrained += 1
		# do we need to train this frame?
		if self._FramesSinceTrained >= 10:
			self.Train()
		return

	def Train(self):
		x, y = self._DataManager.GetXYData(self._XLabels, self._YLabels)

		proccessedX = self._PreProccessX(x)
		proccessedY = self._PreProccessY(y)

		self._Train(proccessedX, proccessedY)
		self._FramesSinceTrained = 0

		# evaluate the model
		# prediction = self._Predict(x)
		# loss = self._LossFunc(y, prediction)
		# accurracy = self._AccurracyFunc(y, prediction)

		# self._Logger.LogDict({
		# 	f"{self._Name}_Loss": loss,
		# 	f"{self._Name}_Accurracy": accurracy
		# })
		return





	def _PreProccessX(self, x):

		proccessX = self._DataManager._JoinColumnsData(x)

		return proccessX

	def _PreProccessY(self, y):
		proccessedY = y
		# proccessedY = self._DataManager._JoinColumnsData(y)
		return proccessedY

	def _Predict(self, x):
		if self._FramesSinceTrained < 0:
			self.Train()

		predicted = None
		return predicted

	def _PostProccess(self, prediction):

		return prediction



	def _Train(self, x, y):
		return
import src.DataManager.DataManager as DataManager
import src.Utils.Metrics.Logger as Logger
import src.DataManager.DataColumnTypes as DCT
import typing
import src.Utils.SharedCoreTypes as SCT
import gymnasium.spaces as spaces
from tensorflow.keras.utils import to_categorical
import numpy as np
from numpy.typing import NDArray


class BasePredictor:

	def __init__(self,
			xLabels:typing.List[DCT.DataColumnTypes],
			yLabels:typing.List[DCT.DataColumnTypes]):

		assert len(xLabels) > 0, "xLabels must have at least one element"
		assert len(yLabels) > 0, "yLabels must have at least one element"

		self._XLabels = xLabels
		self._YLabels = yLabels

		className = self.__class__.__name__.replace("Predictor", "")
		predictionName = "".join([y.name for y in yLabels])
		predictionName = predictionName.replace("DataColumnTypes.", "")
		self._Name = f"{predictionName}_{className}"


		self._DataManager = DataManager.DataManager()
		self._Logger = Logger.Logger()

		self._FramesSinceTrained = -1

		return

	def LoadConfig(self, config:SCT.Config) -> None:
		return

	def Save(self, folderPath:str) -> None:
		return

	def Load(self, folderPath:str) -> None:
		return





	def Predict(self, x:NDArray) -> typing.Tuple[NDArray, NDArray[np.float32]]:

		if self._FramesSinceTrained < 0:
			return None, 0.0

		proccessedX = self._PreProccessX(x)

		predicted = self._Predict(proccessedX)
		proccessedPrediction = self._PostProccess(predicted)

		# todo: confidence
		confidence = np.ones(len(x), dtype=np.float32) * 0.5
		return proccessedPrediction, confidence

	def Observe(self, x, y) -> None:

		if self._FramesSinceTrained >= 0:

			proccessedY = self._PreProccessY(y)

			proccessedX = self._PreProccessX(x)
			predicted = self._Predict(proccessedX)

			error = abs(np.mean(proccessedY - predicted))
			self._Logger.LogDict({
				f"{self._Name}_Validation_Error": error
			})

			self._FramesSinceTrained += 1

		# todo: do we need to train this frame?
		if self._FramesSinceTrained >= 10 or self._FramesSinceTrained < 0:
			self.Train()
		return

	def Train(self) -> None:
		x, y = self._DataManager.GetXYData(self._XLabels, self._YLabels)

		if len(y) == 0 or len(y[0]) < 2:
			return

		proccessedX = self._PreProccessX(x)
		proccessedY = self._PreProccessY(y)

		self._Train(proccessedX, proccessedY)
		self._FramesSinceTrained = 0

		# evaluate the model
		prediction = self._Predict(proccessedX)
		error = abs(np.mean(proccessedY - prediction))

		self._Logger.LogDict({
			f"{self._Name}_Trained_Error": error
		})
		return





	def _PreProccessX(self, x:NDArray) -> NDArray:

		proccessX = self._DataManager._JoinColumnsData(x)

		return proccessX

	def _PreProccessY(self, y:NDArray) -> NDArray:

		# is multi Y label
		if len(self._YLabels) > 1:
			return np.array(y)

		y = y[0]
		proccessed = y

		yLabel = self._YLabels[0]

		if (yLabel == DCT.DataColumnTypes.Terminated or
				yLabel == DCT.DataColumnTypes.Truncated):
			# one hot encode the boolean values
			intBools = [int(i) for i in y]
			proccessed = to_categorical(intBools, num_classes=2)

		elif yLabel == DCT.DataColumnTypes.Reward:
			# todo if reward is clipped then we can one hot encode it
			pass

		elif yLabel == DCT.DataColumnTypes.Action:
			if isinstance(self._DataManager.ActionSpace, spaces.Discrete):
				# one hot encode the action
				proccessed = to_categorical(y, num_classes=self._DataManager.ActionSpace.n)

		elif (yLabel == DCT.DataColumnTypes.CurrentState or
				yLabel == DCT.DataColumnTypes.NextState):
			if isinstance(self._DataManager.ObservationSpace, spaces.Discrete):
				# one hot encode the state
				proccessed = to_categorical(y, num_classes=self._DataManager.ObservationSpace.n)



		return np.array(proccessed)



	def _Predict(self, x:NDArray) -> NDArray:
		if self._FramesSinceTrained < 0:
			self.Train()

		predicted = None
		return predicted



	def _PostProccess(self, prediction:NDArray) -> NDArray:

		# is multi Y label
		if len(self._YLabels) > 1:
			return prediction

		proccessed = prediction

		yLabel = self._YLabels[0]

		if (yLabel == DCT.DataColumnTypes.Terminated or
				yLabel == DCT.DataColumnTypes.Truncated):
			# argmax the one hot encoded boolean values
			intBools = np.argmax(prediction, axis=1)
			proccessed = np.array([bool(i) for i in intBools])

		elif yLabel == DCT.DataColumnTypes.Reward:
			# todo if reward is clipped then we can one hot encode it
			pass

		elif yLabel == DCT.DataColumnTypes.Action:
			if isinstance(self._DataManager.ActionSpace, spaces.Discrete):
				# argmax the one hot encoded action
				proccessed = np.argmax(prediction, axis=1)

		elif (yLabel == DCT.DataColumnTypes.CurrentState or
				yLabel == DCT.DataColumnTypes.NextState):
			if isinstance(self._DataManager.ObservationSpace, spaces.Discrete):
				# argmax the one hot encoded state
				proccessed = np.argmax(prediction, axis=1)

		return np.array([proccessed])



	def _Train(self, x:NDArray, y:NDArray) -> None:
		return
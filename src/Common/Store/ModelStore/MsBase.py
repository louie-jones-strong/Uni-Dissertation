from collections import deque
import tensorflow as tf

class MsBase():
	def __init__(self) -> None:
		return

	def FetchNewestWeights(self, modelKey:str, model:tf.keras.models.Model) -> bool:
		return False

	def PushModel(self, modelKey:str, model:tf.keras.models.Model) -> None:
		return
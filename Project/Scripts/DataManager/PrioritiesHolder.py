import numpy as np
from os import path
from numpy.typing import NDArray

class PrioritiesHolder:
	def __init__(self, capacity:int):
		self._Priorities = np.ones((capacity), dtype=np.float32)

		return

	def Save(self, npyPath:str) ->None:
		np.save(npyPath, self._Priorities)
		return

	def Load(self, npyPath:str) ->None:
		if not path.exists(npyPath):
			return
		self._Priorities = np.load(npyPath)
		return

	def GetMetaData(self) ->dict:
		return {
		}





	def GetPriorities(self) ->NDArray[np.float32]:
		return self._Priorities

	def UpdatePriorities(self, indexs:NDArray[np.int_], priorities:NDArray[np.float32], offset:float=0.1) ->None:

		for i in range(len(indexs)):
			idx = indexs[i]
			priority = priorities[i] + offset

			assert priority is not np.nan, "priority is nan"
			assert priority >= 0.0, f"priority are negative: {priority}"

			self._Priorities[idx] = priority

		# check that all priorities are valid
		assert np.all(self._Priorities >= 0.0), f"Priorities are negative: {self._Priorities}"
		return

	def SetStartPriority(self, index:int) ->None:

		priority = max(self._Priorities.max(), 1.0)
		assert priority is not np.nan, "priority is nan"

		self._Priorities[index] = priority

		assert np.all(self._Priorities >= 0.0), "Priorities are negative"
		return

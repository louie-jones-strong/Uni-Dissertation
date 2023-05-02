import numpy as np
from os import path

class PrioritiesHolder:
	def __init__(self, capacity):
		self._Priorities = np.ones((capacity), dtype=np.float32)

		return

	def Save(self, path):
		np.save(path, self._Priorities)
		return

	def Load(self, path):
		if not path.exists(path):
			return
		self._Priorities = np.load(path)
		return

	def GetMetaData(self):
		return {
		}





	def GetPriorities(self):
		return self._Priorities

	def UpdatePriorities(self, indexs, priorities, offset=0.1):

		for i in range(len(indexs)):
			idx = indexs[i]
			priority = priorities[i] + offset

			assert priority is not np.nan, "priority is nan"
			assert priority >= 0.0, f"priority are negative: {priority}"

			self._Priorities[idx] = priority

		# check that all priorities are valid
		assert np.all(self._Priorities >= 0.0), f"Priorities are negative: {self._Priorities}"
		return

	def SetStartPriority(self, index):

		priority = max(self._Priorities.max(), 1.0)
		assert priority is not np.nan, "priority is nan"

		self._Priorities[index] = priority

		assert np.all(self._Priorities >= 0.0), "Priorities are negative"
		return

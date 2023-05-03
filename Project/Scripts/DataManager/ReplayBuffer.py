#region typing dependencies
from typing import TYPE_CHECKING, Any, Optional, Type, TypeVar

import Utils.SharedCoreTypes as SCT

from numpy.typing import NDArray
if TYPE_CHECKING:
	pass
# endregion

# other imports
from . import PrioritiesHolder
import collections
import numpy as np
from os import path, makedirs
import json

from typing import Any, TypeVar, Optional
from numpy.typing import NDArray
import sys


from Environments.BaseEnv import BaseEnv

#inspired by
# https://github.com/deepmind/dqn_zoo/blob/master/dqn_zoo/replay.py

class ReplayBuffer:
	def __init__(self, capacity:int, env:BaseEnv):

		stateShape:tuple[int, ...]  = env.ObservationSpace.shape
		actionShape:tuple[int, ...] = env.ActionSpace.shape

		stateType   = env.ObservationSpace.dtype
		actionType  = env.ActionSpace.dtype


		self.Capacity = capacity
		self.Count = 0
		self.Current = 0

		self._States        = np.empty((capacity,) + stateShape,  dtype=stateType)
		self._Actions       = np.empty((capacity,) + actionShape, dtype=actionType)
		self._Rewards       = np.empty((capacity),               dtype=np.float32)
		self._NextStates    = np.empty((capacity,) + stateShape,  dtype=stateType)
		self._Terminateds   = np.empty((capacity),               dtype=np.bool_)
		self._Truncateds    = np.empty((capacity),               dtype=np.bool_)
		self._FutureRewards = np.empty((capacity),               dtype=np.float32)

		self._PriorityHolders:dict[str, PrioritiesHolder.PrioritiesHolder] = {}

		return

	def EnsurePriorityHolder(self, key:str) ->None:
		if key not in self._PriorityHolders:
			self._PriorityHolders[key] = PrioritiesHolder.PrioritiesHolder(self.Capacity)
		return

	def Add(self, state:SCT.State, action:SCT.Action, reward:SCT.Reward, nextState:SCT.State, terminateds:bool, truncateds:bool, futureReward:SCT.Reward=-sys.maxsize-1) ->None:

		self._States[self.Current]        = state
		self._Actions[self.Current]       = action
		self._Rewards[self.Current]       = reward
		self._NextStates[self.Current]    = nextState
		self._Terminateds[self.Current]   = terminateds
		self._Truncateds[self.Current]    = truncateds
		self._FutureRewards[self.Current] = futureReward

		for key in self._PriorityHolders:
			self._PriorityHolders[key].SetStartPriority(self.Current)


		self.Current += 1
		self.Count = max(self.Count, self.Current)
		self.Current = self.Current % self.Capacity
		return

	def Sample(self, batchSize:int, priorityKey:Optional[str]=None, priorityScale:float=1.0) -> tuple[NDArray[np.int_], NDArray[Any], NDArray[np.int_], NDArray[np.float32], NDArray[Any], NDArray[np.bool_], NDArray[np.bool_], NDArray[np.float32], NDArray[np.float32]]:
		batchSize = min(batchSize, self.Count)


		if priorityKey is None:
			indexs = np.random.choice(self.Count, batchSize)
			priorities = np.ones((self.Count))

		else:

			self.EnsurePriorityHolder(priorityKey)
			rawPriorities = self._PriorityHolders[priorityKey].GetPriorities()


			scaledPriorities = rawPriorities[:self.Count] ** priorityScale

			probabilities = scaledPriorities / sum(scaledPriorities)
			indexs = np.random.choice(self.Count, batchSize, p=probabilities)

			priorities = rawPriorities[indexs]

		states        = self._States[indexs]
		actions       = self._Actions[indexs]
		rewards       = self._Rewards[indexs]
		nextStates    = self._NextStates[indexs]
		terminateds   = self._Terminateds[indexs]
		truncateds    = self._Truncateds[indexs]
		futureRewards = self._FutureRewards[indexs]

		if None in futureRewards:
			futureRewards = None

		return indexs, states, actions, rewards, nextStates, terminateds, truncateds, futureRewards, priorities

	def UpdatePriorities(self, priorityKey:str, indexs:NDArray[np.int_], priorities:NDArray[np.float32], offset:float=0.1) ->None:

		self._PriorityHolders[priorityKey].UpdatePriorities(indexs, priorities, offset=offset)
		return


	def __len__(self) ->int:
		return self.Count

	def Save(self, folderPath:str) ->None:
		if not path.exists(folderPath):
			makedirs(folderPath)

		metaData = {
			"Capacity": self.Capacity,
			"Count": self.Count,
			"Current": self.Current,
			"PriorityHolders": {}
		}
		for key in self._PriorityHolders:
			metaData["PriorityHolders"][key] = self._PriorityHolders[key].GetMetaData()

		# save meta data as json
		with open(path.join(folderPath, "MetaData.json"), "w") as file:
			json.dump(metaData, file)


		# save core replay data as npy
		np.save(path.join(folderPath, "States.npy"), self._States)
		np.save(path.join(folderPath, "Actions.npy"), self._Actions)
		np.save(path.join(folderPath, "Rewards.npy"), self._Rewards)
		np.save(path.join(folderPath, "NextStates.npy"), self._NextStates)
		np.save(path.join(folderPath, "Terminateds.npy"), self._Terminateds)
		np.save(path.join(folderPath, "Truncateds.npy"), self._Truncateds)
		np.save(path.join(folderPath, "FutureRewards.npy"), self._FutureRewards)

		# save priority holders
		priorityHolderFolder = path.join(folderPath, "PriorityHolders")
		if not path.exists(priorityHolderFolder):
			makedirs(priorityHolderFolder)

		for key in self._PriorityHolders:
			path = path.join(priorityHolderFolder, f"{key}.npy")
			self._PriorityHolders[key].Save(path)
		return

	def Load(self, folderPath:str) ->None:
		if not path.exists(folderPath):
			return

		# load meta data from json
		with open(path.join(folderPath, "MetaData.json"), "r") as file:
			metaData = json.load(file)

		self.Capacity = metaData["Capacity"]
		self.Count = metaData["Count"]
		self.Current = metaData["Current"]
		for key in metaData["PriorityHolders"]:
			self._PriorityHolders[key] = PrioritiesHolder.PrioritiesHolder(self.Capacity, **metaData["PriorityHolders"][key])


		# load core replay data from npy
		self._States = np.load(path.join(folderPath, "States.npy"))
		self._Actions = np.load(path.join(folderPath, "Actions.npy"))
		self._Rewards = np.load(path.join(folderPath, "Rewards.npy"))
		self._NextStates = np.load(path.join(folderPath, "NextStates.npy"))
		self._Terminateds = np.load(path.join(folderPath, "Terminateds.npy"))
		self._Truncateds = np.load(path.join(folderPath, "Truncateds.npy"))
		self._FutureRewards = np.load(path.join(folderPath, "FutureRewards.npy"))


		# load priority holders
		priorityHolderFolder = path.join(folderPath, "PriorityHolders")
		if path.exists(priorityHolderFolder):

			for key in self._PriorityHolders:
				path = path.join(priorityHolderFolder, f"{key}.npy")
				self._PriorityHolders[key].Load(path)

		return

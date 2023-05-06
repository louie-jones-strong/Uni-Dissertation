import json
import sys
from os import makedirs, path
from typing import Optional

import numpy as np
import src.DataManager.PrioritiesHolder as PrioritiesHolder
import src.Utils.SharedCoreTypes as SCT
import src.Environments.BaseEnv as BaseEnv
from numpy.typing import NDArray

# inspired by
# https://github.com/deepmind/dqn_zoo/blob/master/dqn_zoo/replay.py

class ReplayBuffer:
	def __init__(self, capacity:int, env:BaseEnv.BaseEnv):


		self.Capacity = capacity
		self.Count = 0
		self.Current = 0

		stateShape = env.ObservationSpace.shape
		actionShape = env.ActionSpace.shape


		stateShape = SCT.JoinTuples((capacity,), stateShape)
		actionShape = SCT.JoinTuples((capacity,), actionShape)

		stateType = env.ObservationSpace.dtype
		actionType = env.ActionSpace.dtype

		self._States = np.empty(stateShape,  dtype=stateType)
		self._Actions = np.empty(actionShape, dtype=actionType)
		self._Rewards = np.empty((capacity), dtype=np.float32)
		self._NextStates = np.empty(stateShape,  dtype=stateType)
		self._Terminateds = np.empty((capacity), dtype=np.bool_)
		self._Truncateds = np.empty((capacity), dtype=np.bool_)
		self._FutureRewards = np.empty((capacity), dtype=np.float32)

		self._PriorityHolders:dict[str, PrioritiesHolder.PrioritiesHolder] = {}

		return

	def EnsurePriorityHolder(self, key:str) -> None:
		if key not in self._PriorityHolders:
			self._PriorityHolders[key] = PrioritiesHolder.PrioritiesHolder(self.Capacity)
		return

	def Add(self,
			state:SCT.State,
			action:SCT.Action,
			reward:SCT.Reward,
			nextState:SCT.State,
			terminateds:bool,
			truncateds:bool,
			futureReward:SCT.Reward = -sys.maxsize-1) -> None:

		self._States[self.Current] = state
		self._Actions[self.Current] = action
		self._Rewards[self.Current] = reward
		self._NextStates[self.Current] = nextState
		self._Terminateds[self.Current] = terminateds
		self._Truncateds[self.Current] = truncateds
		self._FutureRewards[self.Current] = futureReward

		for key in self._PriorityHolders:
			self._PriorityHolders[key].SetStartPriority(self.Current)


		self.Current += 1
		self.Count = max(self.Count, self.Current)
		self.Current = self.Current % self.Capacity
		return

	def Sample(self,
			batchSize:int,
			priorityKey:Optional[str] = None,
			priorityScale:float = 1.0
			) -> tuple[
				NDArray[np.int_],
				SCT.State_List,
				SCT.Action_List,
				SCT.Reward_List,
				SCT.State_List,
				NDArray[np.bool_],
				NDArray[np.bool_],
				SCT.Reward_List,
				NDArray[np.float32]]:

		batchSize = min(batchSize, self.Count)
		assert batchSize > 1, "batch size must be greater than 1"

		priorities:NDArray[np.float32] = np.ones((self.Count), dtype=np.float32)
		indexs = np.random.choice(self.Count, batchSize)

		if priorityKey is not None:

			self.EnsurePriorityHolder(priorityKey)
			rawPriorities = self._PriorityHolders[priorityKey].GetPriorities()


			scaledPriorities = rawPriorities[:self.Count] ** priorityScale

			probabilities = scaledPriorities / sum(scaledPriorities)
			indexs = np.random.choice(self.Count, batchSize, p=probabilities)

			priorities = rawPriorities[indexs]

		states:SCT.State_List = self._States[indexs]
		actions:SCT.Action_List = self._Actions[indexs]
		rewards:SCT.Reward_List = self._Rewards[indexs]
		nextStates:SCT.State_List = self._NextStates[indexs]
		terminateds:NDArray[np.bool_] = self._Terminateds[indexs]
		truncateds:NDArray[np.bool_] = self._Truncateds[indexs]
		futureRewards:SCT.Reward_List = self._FutureRewards[indexs]

		return indexs, states, actions, rewards, nextStates, terminateds, truncateds, futureRewards, priorities

	def UpdatePriorities(self,
			priorityKey:str,
			indexs:NDArray[np.int_],
			priorities:NDArray[np.float32],
			offset:float = 0.1) -> None:

		self._PriorityHolders[priorityKey].UpdatePriorities(indexs, priorities, offset=offset)
		return


	def __len__(self) -> int:
		return self.Count

	def Save(self, folderPath:str) -> None:
		if not path.exists(folderPath):
			makedirs(folderPath)


		holderConfig = {}
		for key in self._PriorityHolders:
			holderConfig[key] = self._PriorityHolders[key].GetMetaData()

		metaData = {
			"Capacity": self.Capacity,
			"Count": self.Count,
			"Current": self.Current,
			"PriorityHolders": holderConfig
		}

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
			itemPath = path.join(priorityHolderFolder, f"{key}.npy")
			self._PriorityHolders[key].Save(itemPath)
		return

	def Load(self, folderPath:str) -> None:
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
				itemPath = path.join(priorityHolderFolder, f"{key}.npy")
				self._PriorityHolders[key].Load(itemPath)

		return

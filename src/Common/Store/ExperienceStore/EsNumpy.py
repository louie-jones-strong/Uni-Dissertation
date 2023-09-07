import src.Common.Store.ExperienceStore.EsBase as EsBase
import numpy as np
import os
import typing
from typing import Optional, Any
import src.Common.Utils.PathHelper as PathHelper

class EsNumpy(EsBase.EsBase):

	def __init__(self, savePath:str) -> None:
		super().__init__()
		self.SavePath = savePath

		self.TrajectoriesLens = 1

		self.States:Optional[np.ndarray] = None
		self.NextStates:Optional[np.ndarray] = None
		self.Actions:Optional[np.ndarray] = None
		self.Rewards:Optional[np.ndarray] = None
		self.FutureRewards:Optional[np.ndarray] = None
		self.Terminated:Optional[np.ndarray] = None
		self.Truncated:Optional[np.ndarray] = None
		return




	def EmptyTransitionBuffer(self) -> None:

		for i in range(len(self._TransitionBuffer)):
			transition = self.PopTransition()
			state, nextState, action, reward, futureRewards, terminated, truncated = transition

			if state is None:
				continue

			self.States = self._AddValue(self.States, state)
			self.NextStates = self._AddValue(self.NextStates, nextState)
			self.Actions = self._AddValue(self.Actions, action)
			self.Rewards = self._AddValue(self.Rewards, reward)
			self.FutureRewards = self._AddValue(self.FutureRewards, futureRewards)
			self.Terminated = self._AddValue(self.Terminated, terminated)
			self.Truncated = self._AddValue(self.Truncated, truncated)

		self.Save()
		super().EmptyTransitionBuffer()
		return

	def _AddValue(self, npArray:Optional[np.ndarray], value:Any) -> np.ndarray:

		value = np.expand_dims(value, axis=0)
		if npArray is None:
			npArray = value
		else:
			npArray = np.append(npArray, value, axis=0)
		return npArray

	def Save(self) -> None:

		PathHelper.EnsurePathExists(self.SavePath)

		np.save(os.path.join(self.SavePath, "States.npy"), self.States)
		np.save(os.path.join(self.SavePath, "NextStates.npy"), self.NextStates)
		np.save(os.path.join(self.SavePath, "Actions.npy"), self.Actions)
		np.save(os.path.join(self.SavePath, "Rewards.npy"), self.Rewards)
		np.save(os.path.join(self.SavePath, "FutureRewards.npy"), self.FutureRewards)
		np.save(os.path.join(self.SavePath, "Terminated.npy"), self.Terminated)
		np.save(os.path.join(self.SavePath, "Truncated.npy"), self.Truncated)

		return

	def Load(self) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

		self.States = np.load(os.path.join(self.SavePath, "States.npy"))
		self.NextStates = np.load(os.path.join(self.SavePath, "NextStates.npy"))
		self.Actions = np.load(os.path.join(self.SavePath, "Actions.npy"))
		self.Rewards = np.load(os.path.join(self.SavePath, "Rewards.npy"))
		self.FutureRewards = np.load(os.path.join(self.SavePath, "FutureRewards.npy"))
		self.Terminated = np.load(os.path.join(self.SavePath, "Terminated.npy"))
		self.Truncated = np.load(os.path.join(self.SavePath, "Truncated.npy"))

		return self.States, self.NextStates, self.Actions, self.Rewards, self.FutureRewards, self.Terminated, self.Truncated
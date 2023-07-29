import src.Common.Store.ExperienceStore.EsBase as EsBase
import numpy as np
import os
import typing

class EsNumpy(EsBase.EsBase):

	def __init__(self, runPath) -> None:
		super().__init__(runPath)

		self.TrajectoriesLens = 1

		self.States = None
		self.NextStates = None
		self.Actions = None
		self.Rewards = None
		self.FutureRewards = None
		self.Terminated = None
		self.Truncated = None
		return

	def EmptyTransitionBuffer(self) -> None:

		for i in range(len(self._TransitionBuffer)):
			transition = self.PopTransition()
			state, nextState, action, reward, futureRewards, terminated, truncated = transition

			self.States = self.AddValue(self.States, state)
			self.NextStates = self.AddValue(self.NextStates, nextState)
			self.Actions = self.AddValue(self.Actions, action)
			self.Rewards = self.AddValue(self.Rewards, reward)
			self.FutureRewards = self.AddValue(self.FutureRewards, futureRewards)
			self.Terminated = self.AddValue(self.Terminated, terminated)
			self.Truncated = self.AddValue(self.Truncated, truncated)

		self.Save()
		super().EmptyTransitionBuffer()
		return

	def AddValue(self, npArray:np.ndarray, value:any) -> None:

		value = np.expand_dims(value, axis=0)
		if npArray is None:
			npArray = value
		else:
			npArray = np.append(npArray, value, axis=0)
		return npArray

	def Save(self) -> None:

		np.save(os.path.join(self.RunPath, "States.npy"), self.States)
		np.save(os.path.join(self.RunPath, "NextStates.npy"), self.NextStates)
		np.save(os.path.join(self.RunPath, "Actions.npy"), self.Actions)
		np.save(os.path.join(self.RunPath, "Rewards.npy"), self.Rewards)
		np.save(os.path.join(self.RunPath, "FutureRewards.npy"), self.FutureRewards)
		np.save(os.path.join(self.RunPath, "Terminated.npy"), self.Terminated)
		np.save(os.path.join(self.RunPath, "Truncated.npy"), self.Truncated)

		return

	def Load(self) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

		self.States = np.load(os.path.join(self.RunPath, "States.npy"))
		self.NextStates = np.load(os.path.join(self.RunPath, "NextStates.npy"))
		self.Actions = np.load(os.path.join(self.RunPath, "Actions.npy"))
		self.Rewards = np.load(os.path.join(self.RunPath, "Rewards.npy"))
		self.FutureRewards = np.load(os.path.join(self.RunPath, "FutureRewards.npy"))
		self.Terminated = np.load(os.path.join(self.RunPath, "Terminated.npy"))
		self.Truncated = np.load(os.path.join(self.RunPath, "Truncated.npy"))

		return self.States, self.NextStates, self.Actions, self.Rewards, self.FutureRewards, self.Terminated, self.Truncated
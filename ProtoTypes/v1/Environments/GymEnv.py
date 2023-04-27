from . import BaseEnv
import gym
from copy import deepcopy



class GymEnv(BaseEnv.BaseEnv):
	def __init__(self, envConfig, gymEnv=None):
		super().__init__(envConfig)

		self._GymEnv = None
		self._RenderCopy = None

		if gymEnv is None:
			seed = 1234

			kargs = self._Config.get("kwargs", {})
			self._GymEnv = gym.make(self._Config["GymID"], **kargs)
			self._GymEnv.metadata["render_fps"] = 100_000


			self._RenderCopy = gym.make(self._Config["GymID"], render_mode=self._Config["RenderMode"], **kargs)
			self._RenderCopy.metadata["render_fps"] = 100_000

			self._GymEnv.reset(seed=seed)
			self._RenderCopy.reset(seed=seed)

		else:
			self._GymEnv = gymEnv
			self._RenderCopy = None

		self.ObservationSpace = self._GymEnv.observation_space
		self.ActionSpace = self._GymEnv.action_space
		self.RewardRange = self._GymEnv.reward_range

		return


	def Step(self, action):
		"""
		:param action:
		:return: nextState, reward, done
		"""
		if self._Done:
			raise Exception("Environment is done")


		nextState, reward, terminated, truncated, _ = self._GymEnv.step(action)
		if self._RenderCopy is not None:
			self._RenderCopy.step(action)

		self._CurrentFrame += 1


		self._Done = terminated or truncated or self._CurrentFrame >= self._Config["MaxSteps"] - 1




		return nextState, reward, self._Done

	def Clone(self):
		super().Clone()

		newGym = deepcopy(self._GymEnv)
		newEnv = GymEnv(self._Config, gymEnv=newGym)

		return newEnv

	def Reset(self):
		super().Reset()

		state, _ = self._GymEnv.reset()
		if self._RenderCopy is not None:
			self._RenderCopy.reset()

		return state



	def Render(self):
		super().Render()
		return

	def __del__(self):
		super().__del__()

		if self._GymEnv is not None:
			self._GymEnv.close()

		if self._RenderCopy is not None:
			self._RenderCopy.close()
		return
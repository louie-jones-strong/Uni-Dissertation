from . import BaseEnv
import gymnasium as gym
from copy import deepcopy
import random


def WrapGym(wrapperName, gymEnv):

	if wrapperName == "AtariWrapper":

		gymEnv = gym.wrappers.AtariPreprocessing(gymEnv,
			noop_max=0,
			frame_skip=1,
			screen_size=84,
			terminal_on_life_loss=True,
			grayscale_obs=True,
			grayscale_newaxis=False,
			scale_obs=True)

	if wrapperName == "FrameStack":
		gymEnv = gym.wrappers.FrameStack(gymEnv, num_stack=4)

	return gymEnv


class GymEnv(BaseEnv.BaseEnv):
	def __init__(self, envConfig, gymEnv=None):
		super().__init__(envConfig)

		self._GymEnv = None
		self._RenderCopy = None

		if gymEnv is None:

			gymConfig = self._Config.get("GymConfig", {})

			gymId = gymConfig["GymID"]
			kargs = gymConfig.get("kwargs", {})
			wrapper = gymConfig.get("Wrapper", None)



			self._GymEnv = gym.make(gymId, **kargs)
			self._GymEnv = WrapGym(wrapper, self._GymEnv)



			# create a copy of the environment for rendering
			# this is because you cannot copy the env if it has been rendered
			self._RenderCopy = gym.make(gymId, render_mode=gymConfig["RenderMode"], **kargs)


			# make sure both environments are seeded the same
			seed = random.randint(0, 100000)
			self._GymEnv.reset(seed=seed)
			self._RenderCopy.reset(seed=seed)

			# set the render fps to a high number so that it renders as fast as possible
			self._GymEnv.metadata["render_fps"] = 100_000
			self._RenderCopy.metadata["render_fps"] = 100_000

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
		super().Step(action)

		if self._Done:
			raise Exception("Environment is done")


		nextState, reward, terminated, truncated, _ = self._GymEnv.step(action)
		if self._RenderCopy is not None:
			self._RenderCopy.step(action)


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
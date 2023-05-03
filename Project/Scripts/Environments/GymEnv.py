#region typing dependencies
from typing import TYPE_CHECKING, Any, Optional, Type, TypeVar

import Utils.SharedCoreTypes as SCT

from numpy.typing import NDArray
if TYPE_CHECKING:
	pass
# endregion

# other imports

from Environments.BaseEnv import BaseEnv
import gymnasium as gym
from .Wrappers import FrameStack, FireResetEnv
from copy import deepcopy
import random

from typing import Any, TypeVar, Optional


def WrapGym(wrappers:list[str], gymEnv:gym.Env, renderEnv:gym.Env)->tuple[gym.Env, gym.Env]:
	if wrappers is None:
		return gymEnv, renderEnv

	if "FireResetEnv" in wrappers:
		gymEnv = FireResetEnv.FireResetEnv(gymEnv)
		renderEnv = FireResetEnv.FireResetEnv(renderEnv)

	if "Atari" in wrappers:

		gymEnv = gym.wrappers.AtariPreprocessing(gymEnv,
			noop_max=0,
			frame_skip=1,
			screen_size=84,
			terminal_on_life_loss=True,
			grayscale_obs=True,
			grayscale_newaxis=False,
			scale_obs=True)

	if "FrameStack" in wrappers:
		gymEnv = FrameStack.FrameStack(gymEnv, 4)

	return gymEnv, renderEnv




class GymEnv(BaseEnv):
	def __init__(self, envConfig:SCT.Config, gymEnv:Optional[gym.Env]=None):
		super().__init__(envConfig)

		self._RenderCopy = None


		if gymEnv is None:

			gymConfig = self._Config.get("GymConfig", {})

			gymId = gymConfig["GymID"]
			kargs = gymConfig.get("kwargs", {})
			wrappers = gymConfig.get("Wrappers", None)



			self._GymEnv = gym.make(gymId, **kargs)



			# create a copy of the environment for rendering
			# this is because you cannot copy the env if it has been rendered
			self._RenderCopy = gym.make(gymId, render_mode=gymConfig["RenderMode"], **kargs)

			# wrap the environments
			self._GymEnv, self._RenderCopy = WrapGym(wrappers, self._GymEnv, self._RenderCopy)


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


	def Step(self, action:SCT.Action) ->tuple[SCT.State, SCT.Reward, bool, bool]:
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

		self._Done = terminated or truncated

		return nextState, reward, terminated, truncated

	def Clone(self) ->BaseEnv:
		super().Clone()

		newGym = deepcopy(self._GymEnv)
		newEnv = GymEnv(self._Config, gymEnv=newGym)

		return newEnv

	def Reset(self) ->Any:
		super().Reset()

		state, _ = self._GymEnv.reset()
		if self._RenderCopy is not None:
			self._RenderCopy.reset()

		return state



	def Render(self) ->None:
		super().Render()
		return

	def __del__(self) ->None:
		super().__del__()

		if self._GymEnv is not None:
			self._GymEnv.close()

		if self._RenderCopy is not None:
			self._RenderCopy.close()
		return
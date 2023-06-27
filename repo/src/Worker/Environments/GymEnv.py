import random
from copy import deepcopy
from typing import Any, Optional

import gymnasium as gym
import src.Common.Utils.SharedCoreTypes as SCT
import src.Worker.Environments.BaseEnv as BaseEnv
import typing

from src.Worker.Environments.Wrappers import FireResetEnv, FrameStack, ActionDup


def WrapGym(wrappers:typing.List[str], gymEnv:gym.Env, renderEnv:gym.Env) -> typing.Tuple[gym.Env, gym.Env]:
	if wrappers is None:
		return gymEnv, renderEnv

	for wrapper in wrappers:

		if "FireResetEnv" in wrapper:
			gymEnv = FireResetEnv.FireResetEnv(gymEnv)
			renderEnv = FireResetEnv.FireResetEnv(renderEnv)

		elif "Atari" in wrapper:

			gymEnv = gym.wrappers.AtariPreprocessing(gymEnv,
				noop_max=0,
				frame_skip=1,
				screen_size=84,
				terminal_on_life_loss=False,
				grayscale_obs=True,
				grayscale_newaxis=False,
				scale_obs=True)

		elif "FrameStack" in wrapper:
			gymEnv = FrameStack.FrameStack(gymEnv, 4)

		elif "ActionDup" in wrapper:
			gymEnv = ActionDup.ActionDup(gymEnv, 2)
			renderEnv = ActionDup.ActionDup(renderEnv, 2)

		else:
			raise Exception(f"Unknown wrapper: {wrapper}")

	return gymEnv, renderEnv




class GymEnv(BaseEnv.BaseEnv):
	def __init__(self, envConfig:SCT.Config, gymEnv:Optional[gym.Env] = None):
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


		obsSpace = self._GymEnv.observation_space
		assert isinstance(obsSpace, gym.spaces.Discrete) or isinstance(obsSpace, gym.spaces.Box), \
			"obsSpace is not of type Discrete or Box"

		self.ObservationSpace = obsSpace

		actSpace = self._GymEnv.action_space
		assert isinstance(actSpace, gym.spaces.Discrete), \
			"actSpace is not of type Discrete"

		self.ActionSpace = actSpace


		self.RewardRange = self._GymEnv.reward_range

		return


	def Step(self, action:SCT.Action) -> typing.Tuple[SCT.State, SCT.Reward, bool, bool]:
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

		return nextState, float(reward), terminated, truncated

	def Clone(self) -> BaseEnv.BaseEnv:
		super().Clone()

		newGym = deepcopy(self._GymEnv)
		newEnv = GymEnv(self._Config, gymEnv=newGym)

		return newEnv

	def Reset(self) -> Any:
		super().Reset()

		state, _ = self._GymEnv.reset()
		if self._RenderCopy is not None:
			self._RenderCopy.reset()

		return state



	def Render(self) -> None:
		super().Render()
		return
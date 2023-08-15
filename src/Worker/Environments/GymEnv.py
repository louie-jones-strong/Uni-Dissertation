import random
from copy import deepcopy
from typing import Any, Optional

import gymnasium as gym
import src.Common.Utils.SharedCoreTypes as SCT
import src.Worker.Environments.BaseEnv as BaseEnv
import typing

from src.Worker.Environments.Wrappers import FireResetEnv, FrameStack, ActionDup


def WrapGym(
		wrappers:typing.List[str],
		gymEnv:gym.Env,
		rgbRenderEnv:gym.Env,
		humanRenderEnv:gym.Env,
		) -> typing.Tuple[gym.Env, gym.Env]:

	if wrappers is None:
		return gymEnv, rgbRenderEnv, humanRenderEnv

	for wrapper in wrappers:

		if "FireResetEnv" in wrapper:
			gymEnv = FireResetEnv.FireResetEnv(gymEnv)
			rgbRenderEnv = FireResetEnv.FireResetEnv(rgbRenderEnv)
			humanRenderEnv = FireResetEnv.FireResetEnv(humanRenderEnv)

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
			rgbRenderEnv = ActionDup.ActionDup(rgbRenderEnv, 2)
			humanRenderEnv = ActionDup.ActionDup(humanRenderEnv, 2)

		else:
			raise Exception(f"Unknown wrapper: {wrapper}")

	return gymEnv, rgbRenderEnv, humanRenderEnv




class GymEnv(BaseEnv.BaseEnv):
	def __init__(self, envConfig:SCT.Config, gymEnv:Optional[gym.Env] = None):
		super().__init__(envConfig)

		self._RgbRenderCopy = None
		self._HumanRenderCopy = None

		if gymEnv is None:

			gymConfig = self._Config.get("GymConfig", {})

			gymId = gymConfig["GymID"]
			kargs = gymConfig.get("kwargs", {})
			wrappers = gymConfig.get("Wrappers", None)

			self._GymEnv = gym.make(gymId, **kargs)

			# create a copy of the environment for rendering
			# this is because you cannot copy the env if it has been rendered
			self._RgbRenderCopy = gym.make(gymId, render_mode="rgb_array", **kargs)
			self._HumanRenderCopy = gym.make(gymId, render_mode="human", **kargs)

			# wrap the environments
			envs = WrapGym(wrappers, self._GymEnv, self._RgbRenderCopy, self._HumanRenderCopy)
			self._GymEnv, self._RgbRenderCopy, self._HumanRenderCopy = envs


			# make sure both environments are seeded the same
			seed = random.randint(0, 100000)
			self._GymEnv.reset(seed=seed)
			self._RgbRenderCopy.reset(seed=seed)
			self._HumanRenderCopy.reset(seed=seed)

			# set the render fps to a high number so that it renders as fast as possible
			self._GymEnv.metadata["render_fps"] = 100_000
			self._RgbRenderCopy.metadata["render_fps"] = 100_000
			self._HumanRenderCopy.metadata["render_fps"] = 100_000

		else:
			self._GymEnv = gymEnv
			self._RgbRenderCopy = None
			self._HumanRenderCopy = None


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

		if self._RgbRenderCopy is not None:
			self._RgbRenderCopy.step(action)

		if self._HumanRenderCopy is not None:
			self._HumanRenderCopy.step(action)

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

		if self._RgbRenderCopy is not None:
			self._RgbRenderCopy.reset()


		if self._HumanRenderCopy is not None:
			self._HumanRenderCopy.reset()

		return state



	def Render(self) -> Optional[Any]:
		super().Render()

		if self._RgbRenderCopy is None:
			return None

		self._HumanRenderCopy.render()
		rendered:Any = self._RgbRenderCopy.render()
		return rendered
# code was modified from
# https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
#region typing dependencies
from typing import TYPE_CHECKING, Any, Optional, Type, TypeVar

import Utils.SharedCoreTypes as SCT

from numpy.typing import NDArray
if TYPE_CHECKING:
	pass
# endregion

# other file dependencies
import gymnasium as gym
from collections import deque
from gymnasium import spaces
import numpy as np


class FrameStack(gym.Wrapper):
	def __init__(self, env:gym.Env, k:int):
		gym.Wrapper.__init__(self, env)
		self.k = k
		self.frames:deque[SCT.State] = deque([], maxlen=k)

		shp = env.observation_space.shape

		dtype = env.observation_space.dtype
		self.observation_space = spaces.Box(
			low=0,
			high=255,
			shape=(shp + (k,)),
			dtype=dtype)
		return

	def reset(self, **kwargs:Any) -> tuple[NDArray[Any], dict[str, Any]]:
		state, info = self.env.reset(**kwargs)
		for _ in range(self.k):
			self.frames.append(state)
		return self._get_ob(), info

	def step(self, action:SCT.Action) -> tuple[NDArray[Any], SCT.Reward, bool, bool, dict[str, Any]]:
		nextState, reward, terminated, truncated, info = self.env.step(action)
		self.frames.append(nextState)
		return self._get_ob(), reward, terminated, truncated, info

	def _get_ob(self) -> NDArray[Any]:
		assert len(self.frames) == self.k

		stackedFrames = np.array(self.frames)
		stackedFrames = np.swapaxes(stackedFrames, 0, 2)
		return stackedFrames

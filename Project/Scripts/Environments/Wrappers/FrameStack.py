# code was modified from
# https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
import gymnasium as gym
from collections import deque
from gymnasium import spaces
import numpy as np


class FrameStack(gym.Wrapper):
	def __init__(self, env, k):
		"""Stack k last frames.

		Returns lazy array, which is much more memory efficient.

		See Also
		--------
		baselines.common.atari_wrappers.LazyFrames
		"""
		gym.Wrapper.__init__(self, env)
		self.k = k
		self.frames = deque([], maxlen=k)

		shp = env.observation_space.shape

		dtype = env.observation_space.dtype
		self.observation_space = spaces.Box(
			low=0,
			high=255,
			shape=(shp + (k,)),
			dtype=dtype)
		return

	def reset(self, **kwargs):
		state, info = self.env.reset(**kwargs)
		for _ in range(self.k):
			self.frames.append(state)
		return self._get_ob(), info

	def step(self, action):
		nextState, reward, terminated, truncated, info = self.env.step(action)
		self.frames.append(nextState)
		return self._get_ob(), reward, terminated, truncated, info

	def _get_ob(self):
		assert len(self.frames) == self.k

		stackedFrames = np.array(self.frames)
		stackedFrames = np.swapaxes(stackedFrames, 0, 2)
		return stackedFrames

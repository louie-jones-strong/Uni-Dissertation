# code was modified from
# https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py


from typing import Any, SupportsFloat
import Utils.SharedCoreTypes as SCT
from numpy.typing import NDArray

# other file dependencies
import gymnasium as gym
import typing


class ActionDup(gym.Wrapper):
	def __init__(self, env:gym.Env, k:int):
		gym.Wrapper.__init__(self, env)
		self.k = k
		return

	def step(self, action:SCT.Action) -> typing.Tuple[NDArray[Any], SupportsFloat, bool, bool, typing.Dict[str, Any]]:


		totalReward:SupportsFloat = 0.0
		for i in range(self.k):
			nextState, reward, terminated, truncated, info = self.env.step(action)
			totalReward += reward

			if terminated or truncated:
				break

		return nextState, totalReward, terminated, truncated, info

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




class FireResetEnv(gym.Wrapper):
	def __init__(self, env:gym.Env):
		"""Take action on reset for environments that are fixed until firing."""
		gym.Wrapper.__init__(self, env)

		self.NoOp = 0
		self.Fire = 1
		assert env.unwrapped.get_action_meanings()[self.Fire] == 'FIRE'

		actionsNum = env.action_space.n
		self.action_space = gym.spaces.Discrete(actionsNum-1)

		return

	def reset(self, **kwargs) -> tuple[SCT.State, dict[str, Any]]:

		state, info = self.env.reset(**kwargs)
		state, reward, terminated, truncated, info = self.env.step(1)

		return state, info

	def step(self, action:SCT.Action) -> tuple[SCT.State, SCT.Reward, bool, bool, dict[str, Any]]:

		if action >= self.Fire:
			action += 1

		return self.env.step(action)
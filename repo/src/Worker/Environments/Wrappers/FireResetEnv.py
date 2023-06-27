# code was modified from
# https://github.com/openai/baselines/blob/master/baselines/src.Common/atari_wrappers.py

from typing import Any, SupportsFloat
import src.Common.Utils.SharedCoreTypes as SCT
import typing

# other file dependencies
import gymnasium as gym




class FireResetEnv(gym.Wrapper):
	def __init__(self, env:gym.Env):
		"""Take action on reset for environments that are fixed until firing."""
		gym.Wrapper.__init__(self, env)

		self.Fire = 1
		self.LastLives = 0

		# check action space is discrete
		assert isinstance(env.action_space, gym.spaces.Discrete), \
			'This wrapper only works with envs with discrete action spaces (e.g. Breakout)'

		# decrease action space by 1 to account for the fire action
		actionsNum = env.action_space.n
		self.action_space = gym.spaces.Discrete(actionsNum-1)

		return

	def reset(self, **kwargs:Any) -> typing.Tuple[SCT.State, typing.Dict[str, Any]]:

		state, info = self.env.reset(**kwargs)
		state, reward, terminated, truncated, info = self.env.step(self.Fire)
		self.LastLives = info['lives']

		return state, info



	def step(self, action:SCT.Action) -> typing.Tuple[Any, SupportsFloat, bool, bool, typing.Dict[str, Any]]:

		if action >= self.Fire:
			action += 1

		state, reward, terminated, truncated, info = self.env.step(action)

		# if life was lost, then fire to restart the game
		if not (terminated or truncated):

			currentLives = info['lives']
			if currentLives < self.LastLives:
				state, reward, terminated, truncated, info = self.env.step(self.Fire)

			self.LastLives = currentLives


		return state, reward, terminated, truncated, info
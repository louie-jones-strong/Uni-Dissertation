from typing import Any, SupportsFloat
import src.Common.Utils.SharedCoreTypes as SCT
import typing

# other file dependencies
import gymnasium as gym




class CuratedRewards(gym.Wrapper):
	def __init__(self, env:gym.Env, envName:str):
		gym.Wrapper.__init__(self, env)

		self.EnvName = envName

		# FrozenLake
		self.RewardCollected = False

		# CartPole
		self.TargetPosition = -1

		return


	def step(self, action:SCT.Action) -> typing.Tuple[Any, SupportsFloat, bool, bool, typing.Dict[str, Any]]:

		state, reward, terminated, truncated, info = self.env.step(action)

		curatedReward = 0

		if self.EnvName == "FrozenLake":
			if not self.RewardCollected and state == 0:
				self.RewardCollected = True
				curatedReward = 1

		elif self.EnvName == "CartPole":
			cartPos = state[0]
			posError = abs(cartPos - self.TargetPosition)

			maxPosError = 2.4 + abs(self.TargetPosition)

			normPosError = posError / maxPosError

			curatedReward = 1 - normPosError


		else:
			raise Exception("Unknown environment name: " + self.EnvName)


		info["CuratedReward"] = curatedReward
		return state, reward, terminated, truncated, info

	def reset(self, **kwargs:Any) -> typing.Tuple[SCT.State, typing.Dict[str, Any]]:


		state, info = self.env.reset(**kwargs)
		self.RewardCollected = False

		return state, info
# code was modified from
# https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
import gymnasium as gym




class FireResetEnv(gym.Wrapper):
	def __init__(self, env):
		"""Take action on reset for environments that are fixed until firing."""
		gym.Wrapper.__init__(self, env)

		self.NoOp = 0
		self.Fire = 1
		assert env.unwrapped.get_action_meanings()[self.Fire] == 'FIRE'

		actionsNum = env.action_space.n
		self.action_space = gym.spaces.Discrete(actionsNum-1)

		return

	def reset(self, **kwargs):

		state, info = self.env.reset(**kwargs)
		state, reward, terminated, truncated, info = self.env.step(1)

		return state, info

	def step(self, action):

		if action >= self.Fire:
			action += 1

		return self.env.step(action)
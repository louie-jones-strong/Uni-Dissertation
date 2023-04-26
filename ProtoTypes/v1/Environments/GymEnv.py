from . import BaseEnv
import gym



class GymEnv(BaseEnv.BaseEnv):
	def __init__(self, envConfig):
		super().__init__(envConfig)

		kargs = self._Config.get("kwargs", {})
		self._GymEnv = gym.make(self._Config["GymID"], render_mode=self._Config["RenderMode"], **kargs)
		self._GymEnv.metadata["render_fps"] = 100_000

		self.ObservationSpace = self._GymEnv.observation_space
		self.ActionSpace = self._GymEnv.action_space
		return


	def Step(self, action):
		"""
		:param action:
		:return: nextState, reward, done
		"""
		if self._Done:
			raise Exception("Environment is done")


		nextState, reward, terminated, truncated, _ = self._GymEnv.step(action)
		self._CurrentFrame += 1


		self._Done = terminated or truncated or self._CurrentFrame >= self._Config["MaxSteps"] - 1



		return nextState, reward, self._Done

	def PopAction(self):
		raise NotImplementedError

	def Reset(self):
		super().Reset()

		state, _ = self._GymEnv.reset()
		return state



	def Render(self):
		return
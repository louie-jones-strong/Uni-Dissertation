

def GetEnv(envConfig):
	from . import GymEnv

	lookUp = {
		"Gym": GymEnv.GymEnv,
	}

	envType = envConfig["EnvType"]

	if envType not in lookUp:
		raise Exception(f"EnvType \"{envType}\" not found in {lookUp}")
		return None

	env = lookUp[envType](envConfig)
	return env

class BaseEnv:
	def __init__(self, envConfig):
		self.LoadConfig(envConfig)

		self.ObservationSpace = None
		self.ActionSpace = None

		self._CurrentFrame = 0
		self._Done = False
		return

	def LoadConfig(self, envConfig):
		self._Config = envConfig

		return




	def Step(self, action):
		"""
		:param action:
		:return: nextState, reward, done
		"""
		self._CurrentFrame += 1


		state = None
		reward = 0
		done = False
		return state, reward, done

	def PopAction(self):
		self._CurrentFrame -= 1

		return None

	def Reset(self):
		self._CurrentFrame = 0
		self._Done = False

		state = None
		return state



	def Render(self):
		return
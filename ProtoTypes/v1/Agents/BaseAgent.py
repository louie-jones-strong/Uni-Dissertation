

class BaseAgent:
	def __init__(self, env):
		self.Env = env
		return

	def Reset(self):
		return


	def GetAction(self, state):
		actionValues = self.GetActionValues(state)
		if actionValues is not None:
			return actionValues.argmax()

		return None

	def GetActionValues(self, state):
		return None

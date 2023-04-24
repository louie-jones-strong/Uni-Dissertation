from . import BaseAgent
import numpy as np
import keyboard
import time


class HumanAgent(BaseAgent.BaseAgent):

	def GetAction(self, state):
		super().GetAction(state)

		controls = self.EnvConfig["Controls"]
		fps = self.EnvConfig["FPS"]

		action = None
		while action is None:
			for control in controls:
				if keyboard.is_pressed(control):
					action = controls[control]

		time.sleep(1.0/fps)
		return action

import time

import keyboard
import src.Utils.SharedCoreTypes as SCT

from . import BaseAgent


class HumanAgent(BaseAgent.BaseAgent):

	def GetAction(self, state:SCT.State) -> SCT.Action:
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

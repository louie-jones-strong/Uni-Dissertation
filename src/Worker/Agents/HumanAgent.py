import time

import keyboard
import src.Common.Utils.SharedCoreTypes as SCT

import src.Worker.agents.BaseAgent as BaseAgent


class HumanAgent(BaseAgent.BaseAgent):

	def GetAction(self, state:SCT.State) -> SCT.Action:
		super().GetAction(state)

		humanConfig = self.EnvConfig["AgentConfig"]["Human"]
		controls = humanConfig["Controls"]
		fps = humanConfig["FPS"]

		action = None
		while action is None:
			for control in controls:
				if keyboard.is_pressed(control):
					action = controls[control]

		time.sleep(1.0/fps)
		return action, "HumanAgent"

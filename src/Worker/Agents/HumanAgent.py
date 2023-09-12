import time

import keyboard
import src.Common.Utils.SharedCoreTypes as SCT

import src.Worker.Agents.BaseAgent as BaseAgent
import src.Worker.Environments.BaseEnv as BaseEnv
import numpy as np


class HumanAgent(BaseAgent.BaseAgent):

	def GetAction(self, state:SCT.State, env:BaseEnv.BaseEnv) -> SCT.Action:
		super().GetAction(state, env)

		humanConfig = self.EnvConfig["AgentConfig"]["Human"]
		controls = humanConfig["Controls"]
		fps = humanConfig["FPS"]

		action = None
		while action is None:
			for control in controls:
				if keyboard.is_pressed(control):
					action = controls[control]

		time.sleep(1.0/fps)

		actionValues = np.zeros(self.ActionSpace.n).astype(np.float32)
		return action, actionValues, "HumanAgent"

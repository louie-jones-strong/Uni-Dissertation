#region typing dependencies
from typing import TYPE_CHECKING, Any, Optional, Type, TypeVar

import Utils.SharedCoreTypes as SCT

if TYPE_CHECKING:
	pass
# endregion

# other imports

from . import BaseAgent
import numpy as np
import keyboard
import time


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

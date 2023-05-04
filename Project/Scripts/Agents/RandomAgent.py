from typing import Any

import numpy as np
import Utils.SharedCoreTypes as SCT
from Agents.BaseAgent import BaseAgent
from numpy.typing import NDArray


class RandomAgent(BaseAgent):


	def GetAction(self, state:SCT.State) -> Any:
		super().GetAction(state)
		return self.Env.ActionSpace.sample()


	def GetActionValues(self, state:SCT.State) -> NDArray[np.float32]:
		super().GetActionValues(state)

		actions = int(self.Env.ActionSpace.n)
		return np.random.rand(actions).astype(np.float32)
from typing import Any

import numpy as np
import src.Common.Utils.SharedCoreTypes as SCT
from numpy.typing import NDArray
from src.Worker.Agents.BaseAgent import BaseAgent
import src.Worker.Environments.BaseEnv as BaseEnv


class RandomAgent(BaseAgent):


	def GetAction(self, state:SCT.State, env:BaseEnv.BaseEnv) -> Any:
		super().GetAction(state, env)
		return self.ActionSpace.sample(), "RandomAgent"


	def GetActionValues(self, state:SCT.State, env:BaseEnv.BaseEnv) -> NDArray[np.float32]:
		super().GetActionValues(state, env)

		actions = int(self.ActionSpace.n)
		return np.random.rand(actions).astype(np.float32), "RandomAgent"
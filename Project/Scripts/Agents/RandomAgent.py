#region typing dependencies
from typing import TYPE_CHECKING, Any, Optional, Type, TypeVar

import Utils.SharedCoreTypes as SCT

from numpy.typing import NDArray
if TYPE_CHECKING:
	pass
# endregion

# other imports
import numpy as np
from Agents.BaseAgent import BaseAgent


class RandomAgent(BaseAgent):


	def GetAction(self, state:SCT.State) -> Any:
		super().GetAction(state)
		return self.Env.ActionSpace.sample()


	def GetActionValues(self, state:SCT.State) -> NDArray[np.float32]:
		super().GetActionValues(state)

		actions = int(self.Env.ActionSpace.n)
		return np.random.rand(actions).astype(np.float32)
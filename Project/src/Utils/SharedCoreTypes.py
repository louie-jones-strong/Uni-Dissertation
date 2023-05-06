from typing import Any, Optional
from numpy.typing import NDArray
import numpy as np
from gymnasium.spaces import Discrete, Box

# single step
State = NDArray[Any] | int | tuple
Action = int
Reward = float | int
Config = dict[str, Any]

StateSpace = Discrete | Box
ActionSpace = Discrete



# training batch
State_List = NDArray[np.int_ | np.float32]
Action_List = NDArray[np.int_]
Reward_List = NDArray[np.int_ | np.float32]





def JoinTuples(a:Optional[tuple], b:Optional[tuple]) -> tuple:
	joinedTuple:tuple = tuple()
	if a is not None:
		joinedTuple += a

	if b is not None:
		joinedTuple += b

	return joinedTuple

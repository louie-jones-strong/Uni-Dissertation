from typing import Any
from numpy.typing import NDArray
from gymnasium.spaces import Discrete, Box

State = NDArray[Any] | int | tuple
Action = int
Reward = float | int
Config = dict[str, Any]

StateSpace = Discrete | Box
ActionSpace = Discrete

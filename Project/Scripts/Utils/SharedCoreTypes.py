from typing import Any, Type, TypeVar, SupportsFloat
from numpy.typing import NDArray
from gymnasium.spaces import Discrete, Box

State  = NDArray[Any] | int | tuple
Action = int
Reward = SupportsFloat | float | int
Config = dict[str, Any]

StateSpace = Discrete | Box
ActionSpace = Discrete

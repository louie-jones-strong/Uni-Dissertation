## Declaring Types

### Imports
Imports for typing should be done as follows:
```python
#region typing dependencies
from typing import TYPE_CHECKING, Any, Optional, Type, TypeVar

import Utils.SharedCoreTypes as SCT

from numpy.typing import NDArray
if TYPE_CHECKING:
	pass
# endregion

# other file dependencies
import numpy as np
```

example of a type hint:
```python
def ExampleFunction(
		self,
		agent: 'BaseAgent',
		numpyArray: NDArray[np.float32],
		other: Optional[Any] = None,
		) -> tuple[bool, int]:
	"""
	Example function for type hinting
	"""

	return True, 1
```

example of a generic type hint:
```python
T = TypeVar('T')

def ExampleFunction(
		self,
		agent: 'BaseAgent',
		other: Optional[T] = None,
		) -> Optional[T]:
	"""
	Example function for type hinting
	"""

	return other
```

### Running Type Checking
Type checking should be done with the following command:
```cmd
Test.bat
```
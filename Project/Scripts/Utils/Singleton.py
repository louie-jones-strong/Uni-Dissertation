#region typing dependencies
from typing import TYPE_CHECKING, Any, Optional, Type, TypeVar

import Utils.SharedCoreTypes as SCT

if TYPE_CHECKING:
	pass
# endregion

# other file dependencies


class Singleton(object):
	_Instance = None
	def __new__(cls:Type["Singleton"], *args:Any, **kwargs:Any) ->"Singleton":

		if not isinstance(cls._Instance, cls):
			cls._Instance = object.__new__(cls, *args, **kwargs)

		return cls._Instance
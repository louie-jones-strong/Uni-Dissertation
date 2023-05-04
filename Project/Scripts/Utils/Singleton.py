from typing import Any, Type

class Singleton(object):
	_Instance = None

	def __new__(cls:Type["Singleton"], *args:Any, **kwargs:Any) -> 'Singleton':

		if not isinstance(cls._Instance, cls):
			cls._Instance = object.__new__(cls, *args, **kwargs)

		return cls._Instance
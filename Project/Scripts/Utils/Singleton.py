class Singleton(object):
	_Instance = None
	def __new__(cls, *args, **kwargs):

		if not isinstance(cls._Instance, cls):
			cls._Instance = object.__new__(cls, *args, **kwargs)

		return cls._Instance
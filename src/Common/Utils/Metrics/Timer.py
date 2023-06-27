import time
from typing import Callable, Optional, Type
import types

class Timer:
	def __init__(self,
			label:str,
			completeCallback:Optional[Callable[['Timer'], None]] = None
			) -> None:

		self._Label = label
		self._CompleteCallback = completeCallback

		self._Start = 0.0
		self._End = 0.0
		self._Interval = 0.0
		return


	def __enter__(self) -> None:
		self._Start = time.process_time()
		return

	def __exit__(self,
			exc_type:Optional[Type[BaseException]],
			exc_value:Optional[BaseException],
			exc_traceback:Optional[types.TracebackType]) -> None:

		self._End = time.process_time()
		self._Interval = self._End - self._Start

		if self._CompleteCallback is not None:
			self._CompleteCallback(self)
		return
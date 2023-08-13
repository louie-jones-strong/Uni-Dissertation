import datetime


def ConvertNsDuration(ns: int) -> str:
	"""
	Converts a duration in nanoseconds to a human readable format
	:param ns: duration in nanoseconds
	:return: human readable duration
	"""

	ms = ns / 1000000
	s = ms / 1000
	m = s / 60
	h = m / 60


	# if less than 1ms, return ns
	if ms < 1:
		return f"{ns}ns"
	elif s < 1:
		return f"{ms:.2f}ms"
	elif m < 1:
		return f"{s:.2f}s"
	elif h < 1:
		return f"{m:.2f}m"
	else:
		return f"{h:.2f}h"


def ConvertNsTime(ns: int) -> str:
	"""
	Converts a time in nanoseconds to a human readable format
	:param ns: time in nanoseconds
	:return: human readable time
	"""

	date = datetime.datetime.fromtimestamp(ns / 1e9)
	return date.strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]


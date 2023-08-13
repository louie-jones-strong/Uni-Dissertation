from os.path import dirname, abspath, exists
from os import makedirs


def GetRootPath() -> str:
	rootPath = dirname(dirname(dirname(dirname(abspath(__file__)))))
	return rootPath

def EnsurePathExists(path):

	# remove the file part of the path if it is a file
	if "." in path:
		path = dirname(path)

	if not exists(path):
		makedirs(path)
	return
from os.path import dirname, abspath


def GetRootPath() -> str:
	rootPath = dirname(dirname(dirname(abspath(__file__))))
	return rootPath
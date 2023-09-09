import logging


def SetupLogging(level):
	consoleHandler = logging.StreamHandler()
	consoleHandler.setFormatter(CustomFormatter("[%(levelname)s] %(name)s:", styleOn=True))

	fileHandler = logging.FileHandler("log.log", mode="w")
	fileHandler.setFormatter(CustomFormatter("%(asctime)s [%(levelname)s] %(name)s:", styleOn=False))


	logging.basicConfig(level=level, handlers=[consoleHandler, fileHandler])
	return


# ANSI text codes
ANSI_RESET = "\33[0m"
ANSI_BOLD = "\33[31m"

ANSI_BLACK = "\u001b[30m"
ANSI_RED = "\u001b[31m"
ANSI_GREEN = "\u001b[32m"
ANSI_YELLOW = "\u001b[33m"
ANSI_BLUE = "\u001b[34m"
ANSI_MAGENTA = "\u001b[35m"
ANSI_CYAN = "\u001b[36m"
ANSI_WHITE = "\u001b[37m"


class CustomFormatter(logging.Formatter):

	def __init__(self, format, styleOn):
		super().__init__()

		levelStyles = {
			logging.DEBUG: f"{ANSI_GREEN}",
			logging.INFO: f"{ANSI_BLUE}",
			logging.WARNING: f"{ANSI_YELLOW}",
			logging.ERROR: f"{ANSI_RED}",
			logging.CRITICAL: f"{ANSI_MAGENTA}{ANSI_BOLD}"
		}

		self.Formatters = {}

		for logLevel, style in levelStyles.items():

			if styleOn:
				levelFormat = f"{style}{format}{ANSI_RESET} %(message)s"
			else:
				levelFormat = f"{format} %(message)s"

			self.Formatters[logLevel] = logging.Formatter(levelFormat)


		return

	def format(self, record):
		formatter = self.Formatters[record.levelno]
		return formatter.format(record)

import logging


def SetupLogging(level):
	handler = logging.StreamHandler()
	handler.setFormatter(CustomFormatter())

	logging.basicConfig(level=level, handlers=[handler])
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

	def __init__(self):
		super().__init__()

		sharedFormat = "{asctime} [{levelname}] {name}: {message}"

		levelFormats = {
			logging.DEBUG: f"{sharedFormat}{ANSI_RESET}",
			logging.INFO: f"{ANSI_BLUE}{sharedFormat}{ANSI_RESET}",
			logging.WARNING: f"{ANSI_YELLOW}{sharedFormat}{ANSI_RESET}",
			logging.ERROR: f"{ANSI_RED}{sharedFormat}{ANSI_RESET}",
			logging.CRITICAL: f"{ANSI_MAGENTA}{ANSI_BOLD}{sharedFormat}{ANSI_RESET}"
		}

		self.Formatters = {}

		for logLevel, format in levelFormats.items():
			self.Formatters[logLevel] = logging.Formatter(format, style='{')


		return

	def format(self, record):
		formatter = self.Formatters[record.levelno]
		return formatter.format(record)

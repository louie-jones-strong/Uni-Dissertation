import src.Common.Utils.Config.ConfigManager as ConfigManager
import src.Common.Utils.SharedCoreTypes as SCT
import logging

class ConfigurableClass:

	def __init__(self) -> None:
		self.ConfigManager = ConfigManager.ConfigManager()
		return



# region config
	@property
	def Config(self) -> SCT.Config:
		return self.ConfigManager.Config

	@Config.setter
	def Config(self, value:SCT.Config) -> None:
		logging.error("Config is read only", stack_info=True)
		return

# endregion config

# region env config

	@property
	def EnvConfig(self) -> SCT.Config:
		return self.ConfigManager.EnvConfig

	@EnvConfig.setter
	def EnvConfig(self, value:SCT.Config) -> None:
		logging.error("EnvConfig is read only", stack_info=True)
		return

# endregion env config

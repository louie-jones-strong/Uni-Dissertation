import src.Common.Utils.Config.ConfigManager as ConfigManager

class ConfigurableClass:

	def LoadConfig(self) -> None:
		configManager = ConfigManager.ConfigManager()
		self.Config = configManager.Config
		self.EnvConfig = configManager.EnvConfig
		return
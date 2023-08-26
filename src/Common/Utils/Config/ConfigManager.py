import src.Common.Utils.Singleton as Singleton
import src.Common.Utils.Config.ConfigHelper as ConfigHelper
import os


class ConfigManager(Singleton.Singleton):
	def Setup(self, envName:str) -> None:

		mlConfigPath = ConfigHelper.GetClassConfigPath("MLConfig")
		self.Config = ConfigHelper.LoadConfig(mlConfigPath)

		envConfigPath = os.path.join(ConfigHelper.GetRootPath(), "Config", "Envs", envName)
		self.EnvConfig = ConfigHelper.LoadConfig(envConfigPath)
		return

	def SetConfig(self) -> None:

		return

	def SetEnvConfig(self) -> None:
		return
import src.Common.Utils.Singleton as Singleton
import src.Common.Utils.Config.ConfigHelper as ConfigHelper
import os


class ConfigManager(Singleton.Singleton):
	def Setup(self, envName:str) -> None:

		mlConfigPath = ConfigHelper.GetClassConfigPath("MLConfig")
		self.Config = ConfigHelper.LoadConfig(mlConfigPath)

		overrideConfigPath = ConfigHelper.GetClassConfigPath(f"MLConfig-{envName}")
		if os.path.exists(overrideConfigPath):
			overrideConfig = ConfigHelper.LoadConfig(overrideConfigPath)
			self.Config = ConfigHelper.MergeConfig(self.Config, overrideConfig, allowJoining=False)


		envConfigPath = os.path.join(ConfigHelper.GetRootPath(), "Config", "Envs", envName)
		self.EnvConfig = ConfigHelper.LoadConfig(envConfigPath)
		return

	def SetConfig(self) -> None:

		return

	def SetEnvConfig(self) -> None:
		return
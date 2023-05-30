import typing
import os
import json
from src.Utils.PathHelper import GetRootPath
import src.Utils.SharedCoreTypes as SCT

def LoadAndMergeConfig(instance:object, overrideConfig:SCT.Config) -> SCT.Config:

	configPath = GetClassConfigPath(instance)

	baseConfig = {}
	if os.path.exists(configPath):
		baseConfig = LoadConfig(configPath)

	if HasNoneBaseKeys(baseConfig, overrideConfig):
		instanceName = instance.__class__.__name__
		overrideConfig = overrideConfig.get(instanceName, {})

	baseConfig = MergeConfig(baseConfig, overrideConfig)

	return baseConfig


def GetClassConfigPath(instance:object) -> str:

	name = instance.__class__.__name__
	configPath = os.path.join(GetRootPath(), "Config", f"{name}.json")

	return configPath


def LoadConfig(configPath:str) -> SCT.Config:

	if not os.path.exists(configPath):
		raise Exception(f"configPath {configPath} does not exist")

	if not os.path.isfile(configPath):
		raise Exception(f"configPath {configPath} is not a file")

	with open(configPath, "r") as f:
		config = json.load(f)

	return config





def MergeConfig(
		baseConfig:SCT.Config,
		overrideConfig:SCT.Config) -> SCT.Config:

	for key, value in baseConfig.items():

		if key in overrideConfig:
			if isinstance(value, dict):
				MergeConfig(value, overrideConfig[key])
			elif isinstance(value, list):
				raise NotImplementedError()
			else:
				baseConfig[key] = overrideConfig[key]




	return baseConfig

def HasNoneBaseKeys(baseConfig:SCT.Config, overrideConfig:SCT.Config) -> bool:
	hasNoneBaseKeys = False

	for key, value in overrideConfig.items():
		if key not in baseConfig:
			hasNoneBaseKeys = True
			break

	return hasNoneBaseKeys


import argparse
import typing
from typing import Optional
import os
import src.Common.Utils.UserInputHelper as UI
import enum


class ArgParser:

	class ArgType(enum.Enum):
		File = 0
		Bool = 1
		Options = 2
		Enum = 3

	def __init__(self) -> None:
		self.Parser = argparse.ArgumentParser()
		self.ArgSettings:typing.Dict[str, typing.Dict[str, object]] = {}
		self.ParsedArgs:typing.Dict[str, typing.Tuple[str, bool]] = {}

		return

	def _AddOption(self, name:str, helpStr:str, uiLabel:Optional[str], argType:ArgType) -> None:
		self.Parser.add_argument(f"--{name}", type=str, default=None, help=helpStr)

		self.ArgSettings[name] = {
			"type": argType,
			"help": helpStr,
			"name": name,
			"uiLabel": uiLabel
		}
		return

	def AddFilePathOption(self, name:str, helpStr:str, folderPath:str, uiLabel:Optional[str]) -> None:
		self._AddOption(name, helpStr, uiLabel, ArgParser.ArgType.File)
		self.ArgSettings[name]["folderPath"] = folderPath
		return

	def AddBoolOption(self, name:str, helpStr:str, uiLabel:Optional[str]) -> None:
		self._AddOption(name, helpStr, uiLabel, ArgParser.ArgType.Bool)
		return

	def AddOptionsOption(self, name:str, helpStr:str, options:typing.List[str], uiLabel:Optional[str]) -> None:
		self._AddOption(name, helpStr, uiLabel, ArgParser.ArgType.Options)
		self.ArgSettings[name]["options"] = options
		return

	def AddEnumOption(self, name:str, helpStr:str, enumType:enum.Enum, uiLabel:Optional[str]) -> None:
		self._AddOption(name, helpStr, uiLabel, ArgParser.ArgType.Enum)
		self.ArgSettings[name]["enumType"] = enumType
		return

	def _GetArgs(self) -> typing.Dict[str, typing.Tuple[str, bool]]:


		if len(self.ParsedArgs) > 0:
			return self.ParsedArgs

		args = self.Parser.parse_args()
		self.ParsedArgs = {}

		for argName, argInfo in self.ArgSettings.items():
			value = args.__getattribute__(argName)
			validated = False
			self.ParsedArgs[argName] = (value, validated)

		return self.ParsedArgs

	def Get(self, key:str) -> object:
		args = self._GetArgs()

		if key not in args:
			return None



		argSettings = self.ArgSettings[key]

		valueStr, validated = args[key]
		if not validated:
			value = self._ValidateValue(valueStr, argSettings)

			if value is None:
				value = self._GetValue(argSettings)

			self.ParsedArgs[key] = (value, True)

		return value

	def _ValidateValue(self, value:Optional[str], argInfo:typing.Dict[str, object]) -> Optional[object]:

		if value is None:
			return None

		assert isinstance(value, str), f"value ({value}) must be of type str"


		if argInfo["type"] == ArgParser.ArgType.File:
			assert isinstance(argInfo["folderPath"], str), "value must be of type str"

			# check if it is a file
			if not os.path.isfile(value):
				print(f"Invalid file path: {value}, for {argInfo['name']}")
				return None

			# check if it is in the correct folder
			if not value.startswith(argInfo["folderPath"]):
				print(f"File should be in {argInfo['folderPath']}, for {argInfo['name']}")
				return None

		elif argInfo["type"] == ArgParser.ArgType.Bool:
			value = value.lower()
			if value == "true" or value == "t" or value == "false" or value == "f":
				return value == "true" or value == "t"
			else:
				return None

		elif argInfo["type"] == ArgParser.ArgType.Options:
			if not isinstance(argInfo["options"], list):
				raise Exception("Options should be a list")

			if value not in argInfo["options"]:
				print(f"Invalid option: {value}, for {argInfo['name']}")
				return None

		elif argInfo["type"] == ArgParser.ArgType.Enum:

			if not isinstance(argInfo["enumType"], enum.Enum):
				enumType = argInfo["enumType"]
				assert isinstance(enumType, enum.Enum), f"enumType({argInfo}) must be of type enum.Enum"

				members = enumType.__members__

				for key, member in members.items():
					if value == key:
						return member


				print(f"Invalid option: {value}, for {argInfo['name']}")
				return None

		return value

	def _GetValue(self, argInfo:typing.Dict[str, object]) -> object:
		uiLabel = argInfo["uiLabel"]
		helpStr = argInfo["help"]

		assert isinstance(uiLabel, str), "uiLabel must be of type str"
		assert isinstance(helpStr, str), "helpStr must be of type str"

		print()
		print(helpStr)

		if argInfo["type"] == ArgParser.ArgType.File:
			assert isinstance(argInfo["folderPath"], str), "folderPath must be of type str"
			return UI.FilePicker(uiLabel, argInfo["folderPath"])

		elif argInfo["type"] == ArgParser.ArgType.Bool:
			return UI.BoolPicker(uiLabel)

		elif argInfo["type"] == ArgParser.ArgType.Options:
			assert isinstance(argInfo["options"], list), f"options({argInfo}) must be of type str"
			return UI.OptionPicker(uiLabel, argInfo["options"])

		elif argInfo["type"] == ArgParser.ArgType.Enum:
			enumType = argInfo["enumType"]
			assert isinstance(enumType, enum.Enum), f"enumType({argInfo}) must be of type enum.Enum"

			members = enumType.__members__
			key = UI.OptionPicker(uiLabel, list(members.keys()))
			return members[key]


		return None
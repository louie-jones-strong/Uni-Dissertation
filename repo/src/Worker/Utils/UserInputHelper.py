from typing import TypeVar, Optional
import os
import platform
import keyboard
import typing
import argparse


# region arg parser
class ArgParser:

	def __init__(self) -> None:
		self.Parser = argparse.ArgumentParser()
		self.Args:typing.Dict[str, typing.Dict[str, object]] = {}

		return

	def _AddOption(self, name:str, helpStr:str, uiLabel:Optional[str], typeStr:str) -> None:
		self.Parser.add_argument(f"--{name}", type=str, default=None, help=helpStr)

		self.Args[name] = {
			"type": typeStr,
			"help": helpStr,
			"name": name,
			"uiLabel": uiLabel
		}
		return

	def AddFilePathOption(self, name:str, helpStr:str, folderPath:str, uiLabel:Optional[str]) -> None:
		self._AddOption(name, helpStr, uiLabel, "file")
		self.Args[name]["folderPath"] = folderPath
		return

	def AddBoolOption(self, name:str, helpStr:str, uiLabel:Optional[str]) -> None:
		self._AddOption(name, helpStr, uiLabel, "bool")
		return

	def AddOptionsOption(self, name:str, helpStr:str, options:typing.List[str], uiLabel:Optional[str]) -> None:
		self._AddOption(name, helpStr, uiLabel, "options")
		self.Args[name]["options"] = options
		return

	def GetArgs(self) -> typing.Dict[str, object]:

		args = self.Parser.parse_args()
		argsDict = {}

		for argName, argInfo in self.Args.items():
			value = args.__getattribute__(argName)


			value = self._ValidateValue(value, argInfo)

			if value is None:
				value = self._GetValue(argInfo)

			argsDict[argName] = value

		return argsDict

	def _ValidateValue(self, value:Optional[str], argInfo:typing.Dict[str, object]) -> Optional[object]:

		if value is None:
			return None

		assert isinstance(value, str), "value must be of type str"
		assert isinstance(argInfo["folderPath"], str), "value must be of type str"


		if argInfo["type"] == "file":
			# check if it is a file
			if not os.path.isfile(value):
				print(f"Invalid file path: {value}, for {argInfo['name']}")
				return None

			# check if it is in the correct folder
			if not value.startswith(argInfo["folderPath"]):
				print(f"File should be in {argInfo['folderPath']}, for {argInfo['name']}")
				return None

		elif argInfo["type"] == "bool":
			value = value.lower()
			if value == "true" or value == "t" or value == "false" or value == "f":
				return value == "true" or value == "t"
			else:
				return None

		elif argInfo["type"] == "options":
			if not isinstance(argInfo["options"], list):
				raise Exception("Options should be a list")

			if value not in argInfo["options"]:
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

		if argInfo["type"] == "file":
			assert isinstance(argInfo["folderPath"], str), "folderPath must be of type str"
			return FilePicker(uiLabel, argInfo["folderPath"])
		elif argInfo["type"] == "bool":
			return BoolPicker(uiLabel)
		elif argInfo["type"] == "options":
			assert isinstance(argInfo["options"], list), f"options({argInfo}) must be of type str"
			return OptionPicker(uiLabel, argInfo["options"])

		return None
# endregion

# region console input
def FilePicker(label:str, folderPath:str) -> str:
	files = os.listdir(folderPath)
	choice = OptionPicker(label, files)

	return os.path.join(folderPath, choice)

def NumPicker(label:str, minVal:int, maxVal:int) -> int:

	if minVal > maxVal:
		raise Exception("Min value is greater than max value")
	elif minVal == maxVal:
		return minVal

	if label is None or len(label) == 0:
		label = "Pick"

	userInput = input(f"{label}({minVal}-{maxVal}):")

	choice = None

	while True:
		choice = int(userInput)

		if minVal >= 0 and choice <= maxVal:
			break
		else:
			userInput = input("Invalid Please Pick Again:")

	return choice

def BoolPicker(label:str) -> bool:
	if label is None or len(label) == 0:
		label = "Pick"

	userInput = input(f"{label}(True/False):")

	choice:Optional[bool] = None

	while True:
		userInput = userInput.lower()

		if userInput == "true" or userInput == "t":
			choice = True
			break
		elif userInput == "false" or userInput == "f":
			choice = False
			break
		else:
			userInput = input("Invalid Please Pick Again:")

	return choice


T = TypeVar("T")
def OptionPicker(label:str, options:typing.List[T]) -> T:
	if len(options) == 0:
		raise Exception("No options to pick from")
	elif len(options) == 1:
		return options[0]


	for i in range(len(options)):
		print(f"  {i+1}:{options[i]}")

	choice = NumPicker(label, 1, len(options))
	choice -= 1

	return options[choice]
# endregion


# region keyboard input

def IsKeyPressed(key:str) -> bool:

	# check if os is linux
	if platform.system() == "Linux":
		return False

	return keyboard.is_pressed(key)

# endregion
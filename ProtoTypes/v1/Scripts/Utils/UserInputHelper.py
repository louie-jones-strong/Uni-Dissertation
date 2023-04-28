import os

def FilePicker(label, folderPath):
	files = os.listdir(folderPath)
	choice = OptionPicker(label, files)

	return os.path.join(folderPath, choice)

def NumPicker(label, minVal, maxVal, isInt=True):

	if minVal > maxVal:
		raise Exception("Min value is greater than max value")
	elif minVal == maxVal:
		return minVal

	if label is None or len(label) == 0:
		label = "Pick"

	userInput = input(f"{label}({minVal}-{maxVal}): ")

	choice = None

	while True:
		choice = int(userInput)

		if minVal >= 0 and choice <= maxVal:
			break
		else:
			userInput = input("Invalid Please Pick Again: ")

	return choice

def BoolPicker(label):
	if label is None or len(label) == 0:
		label = "Pick"

	userInput = input(f"{label}(True/False): ")

	choice = None

	while True:
		userInput = userInput.lower()

		if userInput== "true" or userInput == "t":
			choice = True
			break
		elif userInput == "false" or userInput == "f":
			choice = False
			break
		else:
			userInput = input("Invalid Please Pick Again: ")

	return choice

def OptionPicker(label, options):
	if len(options) == 0:
		raise Exception("No options to pick from")
	elif len(options) == 1:
		return options[0]


	for i in range(len(options)):
		print(f"  {i+1}: {options[i]}")

	valid = False

	choice = NumPicker(label, 1, len(options), isInt=True)

	choice -= 1

	return options[choice]

import enum

class eModelType(enum.Enum):
	Policy = 0
	Value = 1
	Forward = 2,
	HumanDiscriminator = 3
import enum

class eModelType(enum.Enum):
	Policy = 0
	Value = 1
	PlayStyleDiscriminator = 2,
	Forward = 3,
	Forward_NextState = 4,
	Forward_Reward = 5,
	Forward_Terminated = 6
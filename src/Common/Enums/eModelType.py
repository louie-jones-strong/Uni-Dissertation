import enum

class eModelType(enum.Enum):
	Policy = 0
	Value = 1
	Forward = 2,
	Forward_NextState = 3,
	Forward_Reward = 4,
	Forward_Terminated = 5,
	Human_Discriminator = 6,
	PlayStyle_Discriminator = 7,
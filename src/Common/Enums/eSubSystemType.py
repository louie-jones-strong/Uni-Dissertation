import enum

class eSubSystemType(enum.Enum):
	Learner = 0
	Worker = 1
	Webserver = 2
	ExperienceStore = 3
	ModelStore = 4
	Evaluation = 5
	LatentSpaceHeatmap = 6
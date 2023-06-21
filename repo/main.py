import os

import src.Common.Utils.ArgParser as ArgParser
from src.Common.Enums.AgentType import AgentType
from src.Common.Enums.ModelType import ModelType
from src.Common.Enums.SubSystemType import SubSystemType
from src.Common.Utils.PathHelper import GetRootPath
import src.Common.Utils.ConfigHelper as ConfigHelper


def Main():

	envConfigFolder = os.path.join(GetRootPath(), "Config", "Envs")

	parser = ArgParser.ArgParser()

	parser.AddEnumOption("subsystem", "what sub system is to be ran", SubSystemType, "sub system")
	parser.AddFilePathOption("env", "path to env config", envConfigFolder, "env")

	parser.AddEnumOption("model", "The type of model to train", ModelType, "ModelType")
	parser.AddEnumOption("agent", "agent to use", AgentType, "agent")

	parser.AddBoolOption("play", "Is the agent in training or evaluation?", "playmode")
	# parser.AddBoolOption("wandb", "Should logs be synced to wandb", "wandb sync")
	# parser.AddBoolOption("profile", "Should the runner be profiled", "profile")
	# parser.AddBoolOption("load", "load from previous run", "load")


	# start the subsystem
	subSystem = parser.Get("subsystem")
	envConfigPath = parser.Get("env")
	envConfig = ConfigHelper.LoadConfig(envConfigPath)

	if subSystem == SubSystemType.Learner:
		import src.Learner.Learner as Learner
		learner = Learner.Learner(envConfig, parser.Get("model"))
		learner.Run()

	elif subSystem == SubSystemType.Worker:
		import src.Worker.Worker as Worker
		worker = Worker.Worker(envConfig, parser.Get("agent"), not parser.Get("play"))
		worker.Run()

	elif subSystem == SubSystemType.Webserver:
		import src.WebServer.app as app
		app.Run()

	elif subSystem == SubSystemType.ExperienceStore:
		import src.ExperienceStore.ExperienceStore as ExperienceStore
		ExperienceStore.Run()

	return

if __name__ == "__main__":
	Main()

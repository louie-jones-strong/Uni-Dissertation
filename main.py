import os

import src.Common.Utils.ArgParser as ArgParser
from src.Common.Enums.AgentType import AgentType
from src.Common.Enums.ModelType import ModelType
from src.Common.Enums.SubSystemType import SubSystemType
from src.Common.Utils.PathHelper import GetRootPath
import src.Common.Utils.ConfigHelper as ConfigHelper
import src.Common.Store.ExperienceStore.EsReverb as EsReverb

import time


def Main():

	envConfigFolder = os.path.join(GetRootPath(), "Config", "Envs")

	parser = ArgParser.ArgParser()

	parser.AddEnumOption("subsystem", "what sub system is to be ran", SubSystemType, "sub system")
	parser.AddFilePathOption("env", "path to env config", envConfigFolder, "env")

	parser.AddEnumOption("model", "The type of model to train", ModelType, "ModelType")
	parser.AddEnumOption("agent", "agent to use", AgentType, "agent")

	parser.AddBoolOption("play", "Is the agent in training or evaluation?", "playmode")
	parser.AddBoolOption("wandb", "Should logs be synced to wandb", "wandb sync")
	# parser.AddBoolOption("profile", "Should the runner be profiled", "profile")
	parser.AddBoolOption("load", "load from previous run", "load")


	# get the subsystem settings
	subSystem = parser.Get("subsystem")
	envConfigPath = parser.Get("env")
	envConfig = ConfigHelper.LoadConfig(envConfigPath)


	loggerSubSystemName = None


	if subSystem == SubSystemType.Learner:
		import src.Learner.Learner as Learner

		import src.Common.Utils.ModelHelper as ModelHelper
		modelHelper = ModelHelper.ModelHelper()
		modelHelper.Setup(envConfig)

		model = parser.Get("model")
		load = parser.Get("load")
		learner = Learner.Learner(envConfig, model, load)
		loggerSubSystemName = f"Learner_{model.name}"
		subSystem = learner

	elif subSystem == SubSystemType.Worker:
		import src.Worker.Worker as Worker

		import src.Common.Utils.ModelHelper as ModelHelper
		modelHelper = ModelHelper.ModelHelper()
		modelHelper.Setup(envConfig)

		agent = parser.Get("agent")
		isTrainingMode = not parser.Get("play")


		experienceStore = EsReverb.EsReverb()

		worker = Worker.Worker(envConfig, agent, isTrainingMode, experienceStore)
		loggerSubSystemName = f"Worker_{agent.name}_{'Explore' if isTrainingMode else 'Evaluate'}"

		subSystem = worker

	elif subSystem == SubSystemType.Webserver:
		import src.WebServer.app as app
		subSystem = app

	elif subSystem == SubSystemType.ExperienceStore:
		import src.ExperienceStore.ExperienceStoreSever as ExperienceStoreSever
		subSystem = ExperienceStoreSever



	if loggerSubSystemName is not None:
		envConfig["SubSystemName"] = loggerSubSystemName
		# setup logger
		import src.Common.Utils.Metrics.Logger as Logger
		timeStamp = int(time.time())
		runId = f"{envConfig['Name']}_{loggerSubSystemName}_{timeStamp}"
		runPath = os.path.join(GetRootPath(), "Data", envConfig['Name'])#, runId)

		logger = Logger.Logger()
		logger.Setup(envConfig, runPath, runId=runId, wandbOn=parser.Get("wandb"))


	# run the subsystem
	subSystem.Run()

	return

if __name__ == "__main__":
	Main()

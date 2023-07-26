import os

import src.Common.Utils.ArgParser as ArgParser
from src.Common.Enums.eAgentType import eAgentType
from src.Common.Enums.eModelType import eModelType
from src.Common.Enums.eSubSystemType import eSubSystemType
from src.Common.Utils.PathHelper import GetRootPath
import src.Common.Utils.ConfigHelper as ConfigHelper

import time


def Main():

	envConfigFolder = os.path.join(GetRootPath(), "Config", "Envs")

	parser = ArgParser.ArgParser()

	parser.AddEnumOption("subsystem", "what sub system is to be ran", eSubSystemType, "sub system")
	parser.AddFilePathOption("env", "path to env config", envConfigFolder, "env")

	parser.AddEnumOption("model", "The type of model to train", eModelType, "eModelType")
	parser.AddEnumOption("agent", "agent to use", eAgentType, "agent")

	parser.AddBoolOption("play", "Is the agent in training or evaluation?", "ePlayMode")
	parser.AddBoolOption("wandb", "Should logs be synced to wandb", "wandb sync")
	parser.AddStrOption("rungroup", "grouping for wandb runs", "run group")
	parser.AddBoolOption("load", "load from previous run", "load")


	# get the subsystem settings
	subSystem = parser.Get("subsystem")
	envConfigPath = parser.Get("env")
	envConfig = ConfigHelper.LoadConfig(envConfigPath)


	loggerSubSystemName = None


	if subSystem == eSubSystemType.Learner:
		import src.Learner.Learner as Learner
		import src.Common.Utils.ModelHelper as ModelHelper
		import src.Common.Store.ModelStore.MsRedis as MsRedis

		modelStore = MsRedis.MsRedis()

		modelHelper = ModelHelper.ModelHelper()
		modelHelper.Setup(envConfig, modelStore)

		model = parser.Get("model")
		load = parser.Get("load")
		learner = Learner.Learner(envConfig, model, load)
		loggerSubSystemName = f"Learner_{model.name}"
		subSystem = learner

	elif subSystem == eSubSystemType.Worker:
		import src.Worker.Worker as Worker
		import src.Common.Utils.ModelHelper as ModelHelper
		import src.Common.Store.ModelStore.MsRedis as MsRedis

		modelStore = MsRedis.MsRedis()


		modelHelper = ModelHelper.ModelHelper()
		modelHelper.Setup(envConfig, modelStore)

		agent = parser.Get("agent")
		isTrainingMode = not parser.Get("play")


		import src.Common.Store.ExperienceStore.EsReverb as EsReverb

		experienceStore = EsReverb.EsReverb()

		worker = Worker.Worker(envConfig, agent, isTrainingMode, experienceStore)
		loggerSubSystemName = f"Worker_{agent.name}_{'Explore' if isTrainingMode else 'Evaluate'}"

		subSystem = worker

	elif subSystem == eSubSystemType.Webserver:
		import src.WebServer.app as app
		subSystem = app

	elif subSystem == eSubSystemType.ExperienceStore:
		import src.ExperienceStore.ExperienceStoreServer as ExperienceStoreServer

		subSystem = ExperienceStoreServer.ExperienceStoreServer(envConfig)



	if loggerSubSystemName is not None:
		envConfig["SubSystemName"] = loggerSubSystemName
		envConfig["Group"] = parser.Get("rungroup")

		timeStamp = int(time.time())
		envConfig["RunStartTime"] = timeStamp


		# setup logger
		import src.Common.Utils.Metrics.Logger as Logger
		runId = f"{loggerSubSystemName}_{timeStamp}"
		runPath = os.path.join(GetRootPath(), "Data", envConfig['Name'])#, runId)

		logger = Logger.Logger()
		logger.Setup(envConfig, runPath, runId=runId, wandbOn=parser.Get("wandb"))


	# run the subsystem
	subSystem.Run()

	return

if __name__ == "__main__":
	Main()

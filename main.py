import os

import src.Common.Utils.ArgParser as ArgParser
from src.Common.Enums.eAgentType import eAgentType
from src.Common.Enums.eModelType import eModelType
from src.Common.Enums.eSubSystemType import eSubSystemType
from src.Common.Utils.PathHelper import GetRootPath
import src.Common.Utils.ConfigHelper as ConfigHelper

import time


def CreateExperienceStore(agent, envDataPath, parser):
	experienceStore = None
	if agent == eAgentType.Human:
		import src.Common.Store.ExperienceStore.EsNumpy as EsNumpy

		exampleType = parser.Get("exampleType")
		examplesSavePath = os.path.join(envDataPath, "examples", exampleType)

		experienceStore = EsNumpy.EsNumpy(examplesSavePath)
	else:

		try:
			import src.Common.Store.ExperienceStore.EsReverb as EsReverb
			experienceStore = EsReverb.EsReverb()
		except:
			# used to allow testing on windows
			print("Reverb not installed, using Base Experience Store")
			import src.Common.Store.ExperienceStore.EsBase as EsBase
			experienceStore = EsBase.EsBase()
	return experienceStore

def Main():

	envConfigFolder = os.path.join(GetRootPath(), "Config", "Envs")

	parser = ArgParser.ArgParser()

	parser.AddEnumOption("subsystem", "what sub system is to be ran", eSubSystemType, "sub system")
	parser.AddFilePathOption("env", "path to env config", envConfigFolder, "env")

	parser.AddEnumOption("model", "The type of model to train", eModelType, "ModelType")
	parser.AddEnumOption("agent", "agent to use", eAgentType, "agent")

	parser.AddBoolOption("play", "Is the agent in training or evaluation?", "PlayMode")
	parser.AddBoolOption("wandb", "Should logs be synced to wandb", "wandb sync")
	parser.AddStrOption("rungroup", "grouping for wandb runs", "run group")
	parser.AddBoolOption("load", "load from previous run", "load")
	parser.AddStrOption("exampleType", "type of behaviour example", "example type")


	# get the subsystem settings
	subSystem = parser.Get("subsystem")
	envConfigPath = parser.Get("env")
	envConfig = ConfigHelper.LoadConfig(envConfigPath)


	loggerSubSystemName = None


	envConfig["Group"] = parser.Get("rungroup")
	envDataPath = os.path.join(GetRootPath(), "Data", envConfig['Name'])
	runPath = os.path.join(envDataPath, envConfig["Group"])

	# create the run path
	if not os.path.exists(runPath):
		os.makedirs(runPath)

	if subSystem == eSubSystemType.Learner:
		import src.Learner.Learner as Learner
		import src.Common.Utils.ModelHelper as ModelHelper
		import src.Common.Store.ModelStore.MsRedis as MsRedis

		modelStore = MsRedis.MsRedis()

		modelHelper = ModelHelper.ModelHelper()
		modelHelper.Setup(envConfig, modelStore)

		model = parser.Get("model")
		load = parser.Get("load")

		exampleType = parser.Get("exampleType")
		examplesSavePath = os.path.join(envDataPath, "examples", exampleType)

		learner = Learner.Learner(envConfig, model, load, examplesSavePath)
		loggerSubSystemName = f"Learner_{model.name}"
		subSystem = learner

	elif subSystem == eSubSystemType.Worker:
		import src.Worker.Worker as Worker
		import src.Common.Utils.ModelHelper as ModelHelper
		import src.Common.Store.ModelStore.MsRedis as MsRedis
		import src.Worker.EnvRunner as EnvRunner
		import src.Worker.Environments.BaseEnv as BaseEnv

		modelStore = MsRedis.MsRedis()


		modelHelper = ModelHelper.ModelHelper()
		modelHelper.Setup(envConfig, modelStore)

		agent = parser.Get("agent")

		isTrainingMode = False
		if agent != eAgentType.Human and agent != eAgentType.Random and agent != eAgentType.HardCoded:
			isTrainingMode = not parser.Get("play")

		numEnvs = envConfig["NumEnvsPerWorker"]
		if not isTrainingMode:
			numEnvs = 1
			replayFolder = os.path.join(runPath, "replays")

		envRunners = []
		for i in range(numEnvs):
			env = BaseEnv.GetEnv(envConfig)
			experienceStore = CreateExperienceStore(agent, envDataPath, parser)
			runner = EnvRunner.EnvRunner(env, envConfig["MaxSteps"], experienceStore, replayFolder=replayFolder)
			envRunners.append(runner)

		worker = Worker.Worker(envConfig, agent, envRunners, isTrainingMode)
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

		timeStamp = int(time.time())
		envConfig["RunStartTime"] = timeStamp


		# setup logger
		import src.Common.Utils.Metrics.Logger as Logger
		runId = f"{loggerSubSystemName}_{timeStamp}"

		logger = Logger.Logger()
		logger.Setup(envConfig, runPath, runId=runId, wandbOn=parser.Get("wandb"))


	# run the subsystem
	subSystem.Run()

	return

if __name__ == "__main__":
	Main()

import os

import src.Common.Utils.ArgParser as ArgParser
from src.Common.Enums.eAgentType import eAgentType
from src.Common.Enums.eModelType import eModelType
from src.Common.Enums.eSubSystemType import eSubSystemType
from src.Common.Utils.PathHelper import GetRootPath
import src.Common.Utils.PathHelper as PathHelper
import src.Common.Utils.Config.ConfigHelper as ConfigHelper
import src.Common.Utils.Config.ConfigManager as ConfigManager
import platform

import time


class Main():
	def __init__(self):
		self.Parser = self.DefineCommandLineArgs()
		self.EnvName = self.Parser.Get("env")

		self.ConfigManager = ConfigManager.ConfigManager()
		self.ConfigManager.Setup(self.EnvName)


		self.ConfigManager.EnvConfig["Group"] = self.Parser.Get("rungroup")
		self.EnvDataPath = os.path.join(GetRootPath(), "Data", self.ConfigManager.EnvConfig['Name'])
		self.RunPath = os.path.join(self.EnvDataPath, self.ConfigManager.EnvConfig["Group"])

		PathHelper.EnsurePathExists(self.RunPath)
		return

	def DefineCommandLineArgs(self):
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
		parser.AddBoolOption("saveReplay", "Should the replay be saved", "save replay")
		parser.AddStrOption("exampleType", "type of behaviour example", "example type")

		return parser

#region Setup Subsystems
	def CreateExperienceStore(self, agent):
		experienceStore = None
		if agent == eAgentType.Human:
			import src.Common.Store.ExperienceStore.EsNumpy as EsNumpy

			exampleType = self.Parser.Get("exampleType")
			examplesSavePath = os.path.join(self.EnvDataPath, "examples", exampleType)

			experienceStore = EsNumpy.EsNumpy(examplesSavePath)
		else:

			try:
				import src.Common.Store.ExperienceStore.EsReverb as EsReverb
				experienceStore = EsReverb.EsReverb()
			except:
				# used to allow testing on windows
				# print("Reverb not installed, using Base Experience Store")
				import src.Common.Store.ExperienceStore.EsBase as EsBase
				experienceStore = EsBase.EsBase()
		return experienceStore

	def SetupLogger(self, subSystem, loggerName:str = None):
		self.ConfigManager.EnvConfig["SubSys"] = subSystem

		if loggerName is None:
			loggerName = subSystem

		timeStamp = int(time.time())
		self.ConfigManager.EnvConfig["RunStartTime"] = timeStamp


		# setup logger
		import src.Common.Utils.Metrics.Logger as Logger
		runId = f"{loggerName}_{timeStamp}"

		self.Logger = Logger.Logger()
		self.Logger.Setup(self.ConfigManager.EnvConfig, self.RunPath, runId=runId, wandbOn=self.Parser.Get("wandb"))
		return

	def SetupLearner(self):
		import src.Learner.Learner as Learner
		import src.Common.Utils.ModelHelper as ModelHelper
		import src.Common.Store.ModelStore.MsRedis as MsRedis

		modelStore = MsRedis.MsRedis()

		modelHelper = ModelHelper.ModelHelper()
		modelHelper.Setup(self.ConfigManager.EnvConfig, modelStore)

		model = self.Parser.Get("model")
		load = self.Parser.Get("load")

		exampleType = self.Parser.Get("exampleType")
		examplesSavePath = os.path.join(self.EnvDataPath, "examples", exampleType)

		learner = Learner.Learner(model, load, examplesSavePath)

		self.SetupLogger(f"Learner_{model.name}")
		return learner

	def RunWorker(self, agent:eAgentType, loggerName:str = None, humanRender:bool = True) -> None:
		import src.Worker.Worker as Worker
		import src.Common.Utils.ModelHelper as ModelHelper
		import src.Common.Store.ModelStore.MsBase as MsBase
		import src.Common.Store.ModelStore.MsRedis as MsRedis
		import src.Worker.EnvRunner as EnvRunner
		import src.Worker.Environments.BaseEnv as BaseEnv

		if platform.system() == "Linux":
			modelStore = MsRedis.MsRedis()
		else:
			modelStore = MsBase.MsBase()



		modelHelper = ModelHelper.ModelHelper()
		modelHelper.Setup(self.ConfigManager.EnvConfig, modelStore)


		numEnvs = self.ConfigManager.EnvConfig["NumEnvsPerWorker"]
		if agent == eAgentType.Human or agent == eAgentType.ML:
			numEnvs = 1

		isTrainingMode = False
		if agent != eAgentType.Human and agent != eAgentType.Random and agent != eAgentType.HardCoded:
			isTrainingMode = not self.Parser.Get("play")

			if not isTrainingMode:
				numEnvs = 1

		replayInfo = {
			"Agent": agent.name,
			"IsTrainingMode": isTrainingMode,
			"NumEnvs": numEnvs
		}


		mlConfig = ConfigHelper.LoadConfig(ConfigHelper.GetClassConfigPath("MLConfig"))
		replayInfo = ConfigHelper.FlattenConfig(mlConfig, replayInfo)

		replayFolder = None
		if self.Parser.Get("saveReplay"):
			replayFolder = os.path.join(self.RunPath, "replays", agent.name)


		envRunners = []
		for i in range(numEnvs):
			env = BaseEnv.GetEnv(self.ConfigManager.EnvConfig)
			experienceStore = self.CreateExperienceStore(agent)

			runner = EnvRunner.EnvRunner(env, self.ConfigManager.EnvConfig["MaxSteps"], experienceStore,
				replayFolder=replayFolder, replayInfo=replayInfo, humanRender=False)

			envRunners.append(runner)

		worker = Worker.Worker(self.ConfigManager.EnvConfig, agent, envRunners, isTrainingMode)

		if loggerName is None:
			loggerName = f"Worker_{agent.name}_{'Explore' if isTrainingMode else 'Evaluate'}"

		self.SetupLogger(f"Worker_{agent.name}", loggerName)

		worker.Run()

		self.Logger.Finish()
		print()
		return
#endregion


	def Run(self):
		subSystem = self.Parser.Get("subsystem")

		if subSystem == eSubSystemType.Evaluation:
			self.RunEvaluation()
			return

		elif subSystem == eSubSystemType.Learner:
			subSystem = self.SetupLearner()

		elif subSystem == eSubSystemType.Worker:
			agent = self.Parser.Get("agent")
			self.RunWorker(agent)
			return

		elif subSystem == eSubSystemType.Webserver:
			import src.WebServer.app as app
			subSystem = app.WebServer(self.ConfigManager.EnvConfig)

		elif subSystem == eSubSystemType.ExperienceStore:
			import src.ExperienceStore.ExperienceStoreServer as ExperienceStoreServer
			subSystem = ExperienceStoreServer.ExperienceStoreServer()

		# run the subsystem
		subSystem.Run()

		return

	def RunEvaluation(self):
		evalConfig = ConfigHelper.GetClassConfigPath("EvalConfig")
		evalConfig = ConfigHelper.LoadConfig(evalConfig)

		# set the game limit

		# ============== High Score ==============
		for agentConfig in evalConfig["Agents"]:

			agentTypeStr = agentConfig["AgentType"]
			agentType = eAgentType[agentTypeStr]

			self.ConfigManager.EnvConfig["MaxEpisodes"] = agentConfig["MaxEpisodes"]


			for useRealTime in agentConfig["UseRealSims"]:
				for depth in agentConfig["Depths"]:
					self.ConfigManager.Config["MonteCarloConfig"]["SelectionConfig"]["MaxTreeDepth"] = depth
					self.ConfigManager.Config["UseRealSim"] = useRealTime

					loggerName = f"{agentType.name}_D_{depth}_RT_{useRealTime}"
					self.RunWorker(agentType, loggerName=loggerName, humanRender=False)

		return





if __name__ == "__main__":
	main = Main()
	main.Run()

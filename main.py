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
import logging
import time
import src.Common.Utils.Metrics.LoggingHelper as LoggingHelper


class Main():
	def __init__(self):
		self.EnvConfigFolder = os.path.join(GetRootPath(), "Config", "Envs")

		self.DefineCommandLineArgs()

		logLevel = self.Parser.Get("logLevel")
		LoggingHelper.SetupLogging(logLevel)

		envName = self.Parser.Get("envName")

		self.ConfigManager = ConfigManager.ConfigManager()
		self.ConfigManager.Setup(envName)


		self.ConfigManager.EnvConfig["Group"] = self.Parser.Get("runGroup")
		self.EnvDataPath = os.path.join(GetRootPath(), "Data", self.ConfigManager.EnvConfig['Name'])
		self.RunPath = os.path.join(self.EnvDataPath, self.ConfigManager.EnvConfig["Group"])

		PathHelper.EnsurePathExists(self.RunPath)
		return

	def DefineCommandLineArgs(self) -> None:
		exampleTypes = ["human", "curated"]
		logLevels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

		self.Parser = ArgParser.ArgParser()

		self.Parser.AddEnumOption("subsystem", "what sub system is to be ran", eSubSystemType, "sub system")
		self.Parser.AddFilePathOption("envName", "path to env config", self.EnvConfigFolder, "envName")

		self.Parser.AddEnumOption("model", "The type of model to train", eModelType, "ModelType")
		self.Parser.AddEnumOption("agent", "agent to use", eAgentType, "agent")

		self.Parser.AddBoolOption("play", "Is the agent in training or evaluation?", "PlayMode")
		self.Parser.AddBoolOption("wandb", "Should logs be synced to wandb", "wandb sync")
		self.Parser.AddStrOption("runGroup", "grouping for wandb runs", "run group")
		self.Parser.AddBoolOption("load", "load from previous run", "load")
		self.Parser.AddBoolOption("saveReplay", "Should the replay be saved", "save replay")
		self.Parser.AddOptionsOption("exampleType", "type of behaviour example", exampleTypes, "example type")
		self.Parser.AddOptionsOption("logLevel", "what log level", logLevels, "log Level")


		frameworkConfigPath = ConfigHelper.GetClassConfigPath("FrameworkConfig")
		frameworkConfig = ConfigHelper.LoadConfig(frameworkConfigPath)

		self.Parser.SetDefaults(frameworkConfig)
		return

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
				logging.warning("Reverb not installed, using Base Experience Store")
				import src.Common.Store.ExperienceStore.EsBase as EsBase
				experienceStore = EsBase.EsBase()
		return experienceStore

	def SetupMetrics(self, subSystem, metricName:str = None):
		self.ConfigManager.EnvConfig["SubSys"] = subSystem

		if metricName is None:
			metricName = subSystem

		timeStamp = int(time.time())
		self.ConfigManager.EnvConfig["RunStartTime"] = timeStamp


		# setup metric
		import src.Common.Utils.Metrics.Metrics as Metrics
		runId = f"{metricName}_{timeStamp}"

		self.Metrics = Metrics.Metrics()
		self.Metrics.Setup(self.ConfigManager.EnvConfig, self.RunPath, runId=runId, wandbOn=self.Parser.Get("wandb"))
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


		learner = Learner.Learner(model, load, self.EnvDataPath)

		self.SetupMetrics(f"Learner_{model.name}")
		return learner

	def RunWorker(self, agent:eAgentType, metricName:str = None, humanRender:bool = True) -> None:
		import src.Worker.Worker as Worker
		import src.Common.Utils.ModelHelper as ModelHelper
		import src.Common.Store.ModelStore.MsBase as MsBase
		import src.Common.Store.ModelStore.MsRedis as MsRedis
		import src.Worker.EnvRunner as EnvRunner
		import src.Worker.Environments.BaseEnv as BaseEnv

		numEnvs = self.ConfigManager.EnvConfig["NumEnvsPerWorker"]
		if agent == eAgentType.Human or agent == eAgentType.ML:
			numEnvs = 1

		isTrainingMode = False
		if agent != eAgentType.Human and agent != eAgentType.Random:
			isTrainingMode = not self.Parser.Get("play")

			if not isTrainingMode:
				numEnvs = 1

		if metricName is None:
			if agent == eAgentType.Human:
				exampleType = self.Parser.Get("exampleType")
				metricName = f"Worker_{agent.name}_Example_{exampleType}_Demo"
			else:
				metricName = f"Worker_{agent.name}_{'Explore' if isTrainingMode else 'Evaluate'}"

		if platform.system() == "Linux":
			modelStore = MsRedis.MsRedis()
		else:
			modelStore = MsBase.MsBase()



		modelHelper = ModelHelper.ModelHelper()
		modelHelper.Setup(self.ConfigManager.EnvConfig, modelStore)




		replayInfo = {
			"metricName": metricName,
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
				replayFolder=replayFolder, replayInfo=replayInfo, humanRender=humanRender)

			envRunners.append(runner)

		worker = Worker.Worker(self.ConfigManager.EnvConfig, agent, envRunners, isTrainingMode)


		self.SetupMetrics(f"Worker_{agent.name}", metricName)

		worker.Run()

		self.Metrics.Finish()
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
		evalConfigPath = ConfigHelper.GetClassConfigPath("EvalConfig")
		evalConfig = ConfigHelper.LoadConfig(evalConfigPath)

		playStyles = evalConfig["PlayStyles"]
		agentsConfig = evalConfig["Agents"]

		for playStyle, config in playStyles.items():
			maxEpisodesOverride = config["MaxEpisodes"]

			self.SetPlayStyleConfig(config["normal"], config["human"], config["curated"], config["Temperature"])
			self.EvalStyle(agentsConfig, playStyle, maxEpisodesOverride, humanRender=True)
		return

	def SetPlayStyleConfig(self, normalWeight, humanWeight, curateWeight, temperature):

		# hard coded
		self.ConfigManager.Config["HardcodedConfig"]["PlayStyleWeights"]["Normal"] = normalWeight
		self.ConfigManager.Config["HardcodedConfig"]["PlayStyleWeights"]["Human"] = humanWeight
		self.ConfigManager.Config["HardcodedConfig"]["PlayStyleWeights"]["PlayStyle"] = curateWeight

		self.ConfigManager.Config["MonteCarloConfig"]["NodeScoreConfig"]["RolloutRewardsMultiplier"] = normalWeight
		self.ConfigManager.Config["MonteCarloConfig"]["NodeScoreConfig"]["PlayStyleWeights"]["Human"] = humanWeight
		self.ConfigManager.Config["MonteCarloConfig"]["NodeScoreConfig"]["PlayStyleWeights"]["Curated"] = curateWeight
		self.ConfigManager.Config["MonteCarloConfig"]["ActionSelectionTemperature"] = temperature

		return

	def EvalStyle(self, agentsConfig, evalStyle:str, maxEpisodesOverride:int, humanRender:bool = False) -> None:

		episodes = {}

		for agentConfig in agentsConfig:

			agentTypeStr = agentConfig["AgentType"]
			agentType = eAgentType[agentTypeStr]

			# set the game limit
			self.ConfigManager.EnvConfig["MaxEpisodes"] = min(agentConfig["MaxEpisodes"], maxEpisodesOverride)


			for useRealTime in agentConfig["UseRealSims"]:
				for depth in agentConfig["Depths"]:
					for maxTimePerAction in agentConfig["MaxTimesPerAction"]:
						self.ConfigManager.Config["MonteCarloConfig"]["SelectionConfig"]["MaxTreeDepth"] = depth
						self.ConfigManager.Config["MonteCarloConfig"]["MaxSecondsPerAction"] = maxTimePerAction
						self.ConfigManager.Config["UseRealSim"] = useRealTime

						metricName = f"{agentType.name}_D_{depth}_T_{maxTimePerAction}_RT_{useRealTime}_{evalStyle}"
						self.RunWorker(agentType, metricName=metricName, humanRender=humanRender)

						episodes[metricName] = self.Metrics.EpisodeIds.copy()
						self.Metrics.EpisodeIds.clear()



		episodesPath = os.path.join(self.RunPath, f"{evalStyle}_Episodes.json")
		ConfigHelper.SaveConfig(episodes, episodesPath)

		return





if __name__ == "__main__":
	main = Main()
	main.Run()

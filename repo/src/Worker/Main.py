import os

import Agents.BaseAgent as BaseAgent
import Environments.BaseEnv as BaseEnv
import Utils.UserInputHelper as UI
import Agents.ForwardModel as ForwardModel
from DataManager.DataManager import DataManager
from Utils.Metrics.Logger import Logger
from Utils.PathHelper import GetRootPath
import Utils.ConfigHelper as ConfigHelper
from typing import Optional
import typing

import Runner as Runner
import time

import cProfile
import pstats


def Main(envConfigPath,
		isPlayMode,
		load,
		agentType,
		wandbOn,
		profileOn,
		maxEpisodesOverride:typing.Optional[int] = None,
		maxStepsOverride:typing.Optional[int] = None) -> None:

	# load config
	config = ConfigHelper.LoadConfig(envConfigPath)

	# load agents
	mode = BaseAgent.AgentMode.Train
	if isPlayMode:
		mode = BaseAgent.AgentMode.Play

	# load env
	env = BaseEnv.GetEnv(config)

	timeStamp = int(time.time())
	runId = f"{config['Name']}_{timeStamp}"
	runPath = os.path.join(GetRootPath(), "Data", config['Name'])#, runId)

	# load data manager
	dataManager = DataManager()
	dataManager.Setup(config, env.ObservationSpace, env.ActionSpace, env.RewardRange)


	forwardModel = ForwardModel.ForwardModel(None, config)

	# load logger
	logger = Logger()
	logger.Setup(config, runPath, runId=runId, wandbOn=wandbOn)

	agentConfig = config.get("AgentConfig", {})
	agents = []
	for i in range(1):
		agent = BaseAgent.GetAgent(agentType, agentConfig, mode, forwardModel)
		agents.append(agent)


	# run
	runner = Runner.Runner(envConfigPath, runPath, env, agents, load, forwardModel,
			maxEpisodesOverride=maxEpisodesOverride, maxStepsOverride=maxStepsOverride)

	try:
		if profileOn:
			with cProfile.Profile() as pr:
				runner.RunEpisodes()
		else:
			runner.RunEpisodes()

	except KeyboardInterrupt:
		if UI.BoolPicker("Save?"):
			runner.Save()

	if profileOn:
		stats = pstats.Stats(pr)
		stats.sort_stats(pstats.SortKey.TIME)
		stats.dump_stats(os.path.join(GetRootPath(), "profile.pstats"))

	return


if __name__ == "__main__":

	parser = UI.ArgParser()

	envConfigFolder = os.path.join(GetRootPath(), "Config", "Envs")
	parser.AddFilePathOption("env", "path to env config", envConfigFolder, "env")
	parser.AddBoolOption("play", "run in play mode", "play")
	parser.AddBoolOption("load", "load from previous run", "load")
	parser.AddOptionsOption("agent", "agent to use", BaseAgent.AgentList, "agent")
	parser.AddBoolOption("wandb", "Should logs be synced to wandb", "wandb sync")
	parser.AddBoolOption("profile", "Should the runner be profiled", "profile")

	args = parser.GetArgs()
	print(args)

	try:

		Main(args["env"], args["play"], args["load"], args["agent"], args["wandb"], args["profile"])

	except KeyboardInterrupt:
		print("")
		print("Interrupted")
		os._exit(0)

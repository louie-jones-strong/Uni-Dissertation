import json
import os

import src.Agents.BaseAgent as BaseAgent
import src.Environments.BaseEnv as BaseEnv
import src.Utils.UserInputHelper as UI
import src.Agents.ForwardModel as ForwardModel
from src.DataManager.DataManager import DataManager
from src.Utils.Metrics.Logger import Logger
from src.Utils.PathHelper import GetRootPath
from typing import Optional

import src.Runner as Runner
import time


def Main(envConfigPath,
		isPlayMode,
		load,
		agentType,
		wandbOn) -> None:

	# load config
	with open(envConfigPath) as f:
		config = json.load(f)

	# load agents
	mode = BaseAgent.AgentMode.Train
	if isPlayMode:
		mode = BaseAgent.AgentMode.Play

	# load env
	env = BaseEnv.GetEnv(config)

	timeStamp = int(time.time())
	runId = f"{config['Name']}_{timeStamp}"
	runPath = os.path.join(GetRootPath(), "data", config['Name'])#, runId)

	# load data manager
	dataManager = DataManager()
	dataManager.Setup(config, env.ObservationSpace, env.ActionSpace, env.RewardRange)


	forwardModel = ForwardModel.ForwardModel(None)

	# load logger
	logger = Logger()
	logger.Setup(config, runPath, runId=runId, wandbOn=wandbOn)

	agents = []
	for i in range(1):
		agent = BaseAgent.GetAgent(agentType, config, mode, forwardModel)
		agents.append(agent)


	# run
	runner = Runner.Runner(envConfigPath, runPath, env, agents, load, forwardModel)

	try:
		runner.RunEpisodes()
	except KeyboardInterrupt:
		if UI.BoolPicker("Save?"):
			runner.Save()
	return


if __name__ == "__main__":

	parser = UI.ArgParser()

	envConfigFolder = os.path.join(GetRootPath(), "Config", "Envs")
	parser.AddFilePathOption("env", "path to env config", envConfigFolder, "env")
	parser.AddBoolOption("play", "run in play mode", "play")
	parser.AddBoolOption("load", "load from previous run", "load")
	parser.AddOptionsOption("agent", "agent to use", BaseAgent.AgentList, "agent")
	parser.AddBoolOption("wandb", "Should logs be synced to wandb", "wandb sync")

	args = parser.GetArgs()
	print(args)

	try:

		Main(args["env"], args["play"], args["load"], args["agent"], args["wandb"])

	except KeyboardInterrupt:
		print("")
		print("Interrupted")
		os._exit(0)

# py Main.py --wandb False --env C:\Users\louie\Documents\Git\Uni-Dissertation\Project\Config\Envs\CartPole.json --play False --load False --agent Random
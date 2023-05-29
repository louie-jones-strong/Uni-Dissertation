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


def Main(envIdx:Optional[int],
		isPlayMode:Optional[bool],
		load:Optional[bool],
		agentIdx:Optional[int],
		wandbOn:Optional[bool]) -> None:

	# find all environments in the configs folder
	configPath = os.path.join(GetRootPath(), "Config", "Envs")
	envConfigPath = UI.FilePicker("Environments", configPath)

	# load config
	with open(envConfigPath) as f:
		config = json.load(f)

	# load agents
	mode = BaseAgent.AgentMode.Train
	if UI.BoolPicker("Play"):
		mode = BaseAgent.AgentMode.Play

	load = UI.BoolPicker("Load")
	agentType = UI.OptionPicker(f"Agent", BaseAgent.AgentList)

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

	try:

		Main(None, None, None, None, False)

	except KeyboardInterrupt:
		print("")
		print("Interrupted")
		os._exit(0)

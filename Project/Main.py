import json
import os

import src.Agents.BaseAgent as BaseAgent
import src.Environments.BaseEnv as BaseEnv
import src.Utils.UserInputHelper as UI
from src.DataManager.DataManager import DataManager
from src.Utils.Metrics.Logger import Logger
from src.Utils.PathHelper import GetRootPath

import src.Runner as Runner
import time


def Main() -> None:
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

	agents = []
	for i in range(1):
		agents.append(BaseAgent.GetAgent(agentType)(env.ObservationSpace, env.ActionSpace, config, mode=mode))

	timeStamp = int(time.time())
	runId = f"{config['Name']}_{timeStamp}"
	runPath = os.path.join(GetRootPath(), "data", config['Name'])#, runId)




	# load data manager
	dataManager = DataManager()
	dataManager.Setup(config, env.ObservationSpace, env.ActionSpace)

	# load logger
	logger = Logger()
	logger.Setup(config, runPath, runId=runId)

	# run
	runner = Runner.Runner(envConfigPath, runPath, env, agents, load)

	try:
		runner.RunEpisodes()
	except KeyboardInterrupt:
		if UI.BoolPicker("Save?"):
			runner.Save()


	return


if __name__ == "__main__":

	try:

		Main()

	except KeyboardInterrupt:
		print("")
		print("Interrupted")
		os._exit(0)

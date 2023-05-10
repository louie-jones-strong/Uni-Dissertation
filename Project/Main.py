import json
import os

import src.Agents.BaseAgent as BaseAgent
import src.Environments.BaseEnv as BaseEnv
import src.Utils.UserInputHelper as UI
from src.DataManager.DataManager import DataManager
from src.Utils.PathHelper import GetRootPath

import src.Runner as Runner


def Main() -> None:
	# find all environments in the configs folder
	configPath = os.path.join(GetRootPath(), "Config", "Envs")
	envConfigPath = UI.FilePicker("Environments", configPath)

	# load config
	with open(envConfigPath) as f:
		config = json.load(f)

	# load env
	env = BaseEnv.GetEnv(config)

	# load data manager
	dataManager = DataManager()
	dataManager.Setup(config, env)

	# load agents
	numAgents = UI.NumPicker("Number of agents", 1, 1)

	mode = BaseAgent.AgentMode.Train
	if UI.BoolPicker("Play"):
		mode = BaseAgent.AgentMode.Play

	load = UI.BoolPicker("Load")

	agents = []
	for i in range(numAgents):
		agentType = UI.OptionPicker(f"Agent_{i+1}", BaseAgent.AgentList)
		agent = BaseAgent.GetAgent(agentType)(env, config, mode=mode)
		agents.append(agent)

	# run
	runner = Runner.Runner(envConfigPath, env, agents, load)

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

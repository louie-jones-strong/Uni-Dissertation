import os
import Runner
import Utils.UserInputHelper as UI



def Main():

	# find all environments in the configs folder
	configPath = os.path.join(os.path.abspath(os.curdir), "Config", "Envs")

	envConfigPath = UI.FilePicker("Environments", configPath)

	runner = Runner.Runner(envConfigPath)

	try:
		runner.RunEpisodes(numEpisodes=1000)
	except KeyboardInterrupt:
		print('Interrupted')
		os._exit(0)

	return


if __name__ == "__main__":
	Main()

import src.Common.Agents.BaseAgent as BaseAgent
import src.Common.Utils.SharedCoreTypes as SCT
import src.Worker.EnvRunner as EnvRunner
from src.Common.Enums.eAgentType import eAgentType
import typing
import time


class Worker:
	"""
	Worker is responsible for collecting trajectories to fill the experience store.
	"""

	def __init__(self, envConfig:SCT.Config,
		eAgentType:eAgentType,
		envRunners:typing.List[EnvRunner.EnvRunner],
		isTrainingMode:bool) -> None:

		self.Config = envConfig
		self.IsEvaluating = not isTrainingMode


		self.Agent = BaseAgent.GetAgent(eAgentType, envConfig, isTrainingMode)

		self.Envs = envRunners

		self.EpisodeCount = 0

		self._ModelUpdateTime = time.time() + self.Config["SecsPerModelFetch"]
		return

	def Run(self) -> None:
		"""
		Runs the worker's main loop.
		collecting observations from the envs and sending them to the agent, to get actions.
		then making the actions in the envs.
		"""

		# get initial states from the environments
		stateList = [env.GetState() for env in self.Envs]

		maxEpisodes = self.Config["MaxEpisodes"]

		# run the environments
		while self.EpisodeCount < maxEpisodes:

			# get actions from the agent
			actions, actionReasons = self._GetActions(stateList)

			# make the chosen actions in the environments
			stateList, finishedEpisodes = self._StepEnvs(actions, actionReasons)

			# increment the episode count by the number of episodes that have been completed in this step
			self.EpisodeCount += finishedEpisodes

			if finishedEpisodes > 0:
				print(f"{self.EpisodeCount+1} / {maxEpisodes}")

			# If in evaluate mode then we only check for updates after an episode.
			# This is to ensure that the agent is consistent for the whole episode.
			if not self.IsEvaluating or finishedEpisodes > 0:
				self.CheckForUpdates()

		print("Worker finished")
		return

	def _GetActions(self, stateList:typing.List[SCT.State]) -> typing.Tuple[typing.List[SCT.Action], typing.List[str]]:
		actions = []
		actionReasons = []

		for state in stateList:
			action, actionReason = self.Agent.GetAction(state)
			actions.append(action)
			actionReasons.append(actionReason)

		return actions, actionReasons

	def _StepEnvs(self, actions:typing.List[SCT.Action], actionReasons) -> typing.Tuple[typing.List[SCT.State], int]:
		"""
		Makes the chosen actions in the environments.

		Args:
			actions: list of actions to make in the environments.

		Returns:
			list of states after the actions have been made.
			count the number of episodes that have been completed, in this step.
		"""

		assert len(actions) == len(self.Envs), \
			f"the number of actions must match the number of envs. {len(actions)} != {len(self.Envs)}"


		stateList = []
		finishedEpisodes = 0
		for i in range(len(self.Envs)):

			state, done = self.Envs[i].Step(actions[i], actionReason=actionReasons[i])

			stateList.append(state)

			if done:
				finishedEpisodes += 1

		return stateList, finishedEpisodes


	def CheckForUpdates(self) -> None:
		"""
		Checks if the agent needs to update its models.
		This is based on the time since the last update.
		The time between updates is set in the config.
		"""

		if time.time() >= self._ModelUpdateTime:
			self.Agent.UpdateModels()

			self._ModelUpdateTime = time.time() + self.Config["SecsPerModelFetch"]
		return
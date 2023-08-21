import src.Worker.Agents.BaseAgent as BaseAgent
import numpy as np
import src.Common.Utils.SharedCoreTypes as SCT
from numpy.typing import NDArray
import time
from gymnasium.spaces import Discrete, Box
import src.Worker.Agents.Models.ForwardModel as ForwardModel
import src.Worker.Agents.Models.ValueModel as ValueModel
import src.Worker.Agents.Models.PlayStyleModel as PlayStyleModel
import src.Worker.Agents.TreeNode as TreeNode
import src.Worker.Environments.BaseEnv as BaseEnv
import typing

def AreStatesEqual(state1:SCT.State, state2:SCT.State) -> bool:

	if isinstance(state1, int):
		return state1 == state2

	return np.array_equal(state1, state2)



class MonteCarloAgent(BaseAgent.BaseAgent):

	def __init__(self, envConfig:SCT.Config, isTrainingMode:bool,
				forwardModel:ForwardModel.ForwardModel,
				valueModel:ValueModel.ValueModel,
				playStyleModel:PlayStyleModel.PlayStyleModel):

		self._ForwardModel = forwardModel
		self._ValueModel = valueModel
		self._PlayStyleModel = playStyleModel

		super().__init__(envConfig, isTrainingMode)

		self._CachedTree = None
		self._StopTime = 0
		return

	def UpdateModels(self) -> None:
		super().UpdateModels()
		self._ForwardModel.UpdateModels()
		self._ValueModel.UpdateModels()
		self._PlayStyleModel.UpdateModels()
		return

	def GetAction(self, state:SCT.State, env:BaseEnv.BaseEnv) -> typing.Tuple[SCT.Action, str]:
		super().GetAction(state, env)
		actionValues, actionReason = self.GetActionValues(state, env)
		return BaseAgent.BaseAgent._GetMaxValues(actionValues), actionReason

	def GetActionValues(self, state:SCT.State, env:BaseEnv.BaseEnv) -> typing.Tuple[NDArray[np.float32], str]:
		super().GetActionValues(state, env)

		monteCarloConfig = self.Config["MonteCarloConfig"]

		if not self._ForwardModel.CanPredict():
			return self.ActionSpace.sample(), "MonteCarloAgent Random Action (Can't Predict)"

		self._StopTime = time.process_time() + monteCarloConfig["MaxSecondsPerAction"]

		rootNode = self._CachedTree
		if rootNode is None or not AreStatesEqual(rootNode.State, state):

			if rootNode is not None:
				for child in rootNode.Children:

					if AreStatesEqual(child.State, state):
						rootNode = child
						break

			if rootNode is None or not AreStatesEqual(rootNode.State, state):
				self._CachedTree = None
				rootNode = TreeNode.TreeNode(state, env, self.StepNum, done=False,
					valueModel=self._ValueModel, playStyleModel=self._PlayStyleModel)




		if rootNode.Children is None:
			rootNode.Expand(self.ActionList, self._ForwardModel, self._ValueModel, self._PlayStyleModel)


		selectionConfig = monteCarloConfig["SelectionConfig"]

		exploreFactor = selectionConfig["TestExploreFactor"]
		if self.Mode == BaseAgent.ePlayMode.Train:
			exploreFactor = selectionConfig["TrainExploreFactor"]


		# monte carlo tree search
		for i in range(selectionConfig["MaxSelectionsPerAction"]):

			# 1. selection
			selectedNode = rootNode.Selection(exploreFactor)

			# 2. expansion
			if selectedNode.Counts > 0 and selectedNode.Children is None:
				selectedNode.Expand(self.ActionList, self._ForwardModel, self._ValueModel, self._PlayStyleModel)
				selectedNode = selectedNode.Selection(exploreFactor)

			# 3. simulation
			rolloutMaxDepth = self.EnvConfig["MaxSteps"] - selectedNode.EpisodeStep
			rolloutMaxDepth = min(rolloutMaxDepth, monteCarloConfig["RollOutConfig"]["MaxRollOutDepth"])

			totalRewards = self._RollOut(selectedNode.State, selectedNode.Env, rolloutMaxDepth)

			# 4. backpropagation
			selectedNode.BackPropagate(totalRewards.sum(), len(totalRewards))



			if time.process_time() >= self._StopTime and \
				rootNode.AllExplored():
				break

		if monteCarloConfig["CacheTree"]:
			self._CachedTree = rootNode

		if rootNode.Children is None:
			return self.ActionSpace.sample(), "MonteCarloAgent Random Action (No Children)"


		actionValues, valueModelValues = rootNode.GetActionValues()
		actionReason = {
			"Type": "MonteCarloAgent",
			"ActionValues": actionValues,
			"ValueModelValues": valueModelValues,
			"Tree": rootNode.ToDict()
		}
		return actionValues, actionReason

	def _RollOut(self, state:SCT.State, env:BaseEnv.BaseEnv, maxDepth:int) -> SCT.Reward_List:
		rollOutConfig = self.Config["MonteCarloConfig"]["RollOutConfig"]
		numRollOuts = rollOutConfig["MaxRollOutCount"]
		timeOutAllowed = rollOutConfig["TimeOutAllowed"]


		states, envs = TreeNode.TreeNode.CloneState(state, env, numRollOuts, self.Config["UseRealSim"])

		isPlayingMask = np.ones(numRollOuts, dtype=np.bool_)
		totalRewards = np.zeros(numRollOuts, dtype=np.float32)

		for d in range(maxDepth):

			# get the actions
			actions = self._SampleActions(self.ActionSpace, numRollOuts)

			# predict the next states and rewards
			nextStates, envs, rewards, terminateds = self._ForwardModel.Predict(states, envs, actions)


			totalRewards += rewards * isPlayingMask
			isPlayingMask = np.logical_or(isPlayingMask, terminateds)

			# update the states
			states = nextStates

			if timeOutAllowed and time.process_time() < self._StopTime:
				break

		if rollOutConfig["ValueFinalStates"] and self._ValueModel.CanPredict():
			values = self._ValueModel.Predict(states)
			totalRewards += values * isPlayingMask

		return totalRewards


	def _SampleActions(self, actionSpace: SCT.ActionSpace, numActions:int) -> SCT.Action_List:

		if isinstance(actionSpace, Discrete):
			n = actionSpace.n
			return np.random.randint(n, size=(numActions,))

		elif isinstance(actionSpace, Box):
			low = actionSpace.low
			high = actionSpace.high
			return np.random.uniform(low, high, size=(numActions,))

		else:
			raise NotImplementedError("Unknown action space type: " + str(type(actionSpace)))

		return

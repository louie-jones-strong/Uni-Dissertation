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

	def __init__(self, isTrainingMode:bool,
				forwardModel:ForwardModel.ForwardModel,
				valueModel:ValueModel.ValueModel,
				playStyleModels:typing.Dict[str, PlayStyleModel.PlayStyleModel]):

		self._ForwardModel = forwardModel
		self._ValueModel = valueModel
		self._PlayStyleModels = playStyleModels

		super().__init__(isTrainingMode)

		self._CachedTree = None
		self._StopTime = 0
		return

	def UpdateModels(self) -> None:
		super().UpdateModels()
		self._ForwardModel.UpdateModels()
		self._ValueModel.UpdateModels()

		for model in self._PlayStyleModels.values():
			model.UpdateModels()
		return

	def Reset(self) -> None:
		super().Reset()

		if not self.Config["MonteCarloConfig"]["CacheTreeBetweenEpisodes"]:
			self._CachedTree = None

		return

	def GetAction(self, state:SCT.State, env:BaseEnv.BaseEnv) -> \
			typing.Tuple[SCT.Action, SCT.ActionValues, SCT.ActionReason]:

		super().GetAction(state, env)
		actionValues, actionReason = self.GetActionValues(state, env)

		temperature = self.Config["MonteCarloConfig"]["ActionSelectionTemperature"]

		action, probabilities = BaseAgent.BaseAgent._SoftMaxSelection(actionValues, temperature)
		actionReason["ActionProbabilities"] = probabilities

		maxDepth = 0
		if "Tree" in actionReason:
			treeNode = actionReason["Tree"]
			maxDepth = treeNode["MaxDepth"] - self.StepNum

		actionValuesStr = str(np.round(actionValues, 2))
		probabilitiesStr = str(np.round(probabilities, 2))
		self.Logger.debug(f"Action: {action} {actionValuesStr} {probabilitiesStr} Depth: {maxDepth}")

		return action, actionValues, actionReason

	def GetActionValues(self, state:SCT.State, env:BaseEnv.BaseEnv) -> \
			typing.Tuple[NDArray[np.float32], SCT.ActionReason]:

		super().GetActionValues(state, env)

		monteCarloConfig = self.Config["MonteCarloConfig"]

		if not self._ForwardModel.CanPredict():
			return np.ones(self.ActionSpace.n), {"Reason": "MonteCarloAgent Random Action (Can't Predict)"}

		self._StopTime = time.process_time() + monteCarloConfig["MaxSecondsPerAction"]

		stateNode = self.GetCurrentStateNode()
		if stateNode is None or not AreStatesEqual(stateNode.State, state):
			self._CachedTree = None
			stateNode = TreeNode.TreeNode(state, env, self.StepNum, done=False,
				valueModel=self._ValueModel, playStyleModels=self._PlayStyleModels)




		if stateNode.Children is None:
			stateNode.Expand(self.ActionList, self._ForwardModel, self._ValueModel, self._PlayStyleModels)


		selectionConfig = monteCarloConfig["SelectionConfig"]

		exploreFactor = selectionConfig["TestExploreFactor"]
		if self.Mode == BaseAgent.ePlayMode.Train:
			exploreFactor = selectionConfig["TrainExploreFactor"]

		maxTreeDepth = self.StepNum + selectionConfig["MaxTreeDepth"]
		maxTreeDepth = min(maxTreeDepth, self.EnvConfig["MaxSteps"])

		# monte carlo tree search
		for i in range(selectionConfig["MaxSelectionsPerAction"]):

			# 1. selection
			with self._Metrics.Time("Selection"):
				selectedNode = stateNode.Selection(exploreFactor, maxTreeDepth)



			# 2. expansion
			if selectedNode is not None and selectedNode.Counts > 0 and selectedNode.Children is None:
				with self._Metrics.Time("Expansion"):
					selectedNode.Expand(self.ActionList, self._ForwardModel, self._ValueModel, self._PlayStyleModels)
					selectedNode = selectedNode.Selection(exploreFactor, maxTreeDepth)

			if selectedNode is None:
				break

			# 3. simulation
			with self._Metrics.Time("RollOut"):
				totalRewards = self._RollOut(selectedNode)

			# 4. backpropagation
			selectedNode.BackPropRewards(totalRewards.sum(), len(totalRewards))



			if time.process_time() >= self._StopTime and \
				stateNode.AllExplored():
				break

		if monteCarloConfig["CacheTreeBetweenActions"] and self._CachedTree is None:
			self._CachedTree = stateNode

		if stateNode.Children is None:
			return np.ones(self.ActionSpace.n), {"Reason": "MonteCarloAgent Random Action (No Children)"}




		actionValues = stateNode.GetActionValues()
		actionReason = {
			"Type": "MonteCarloAgent",
			"ActionValues": actionValues,
			"Tree": stateNode.ToDict()
		}
		return actionValues, actionReason

	@staticmethod
	def _SampleActions(actionSpace: SCT.ActionSpace, numActions:int) -> SCT.Action_List:

		if isinstance(actionSpace, Discrete):
			n = actionSpace.n
			# create array with each action
			samples = np.arange(start=0, stop=n, step=1, dtype=np.int32)

			# repeat the array to get the number of actions
			repeats = numActions / n
			if repeats > 1:

				# check if reapeats is an integer
				if repeats % 1 == 0:
					samples = np.repeat(samples, repeats)
				else:
					samples = np.repeat(samples, (numActions // n) + 1)
					samples = samples[:numActions]

			# randomly shuffle the array
			np.random.shuffle(samples)


			return samples

		elif isinstance(actionSpace, Box):
			low = actionSpace.low
			high = actionSpace.high
			return np.random.uniform(low, high, size=(numActions,))

		else:
			raise NotImplementedError("Unknown action space type: " + str(type(actionSpace)))

		return

	def GetCurrentStateNode(self) -> TreeNode.TreeNode:
		stateNode = self._CachedTree

		for action in self.ActionHistory:
			if stateNode is None:
				break

			childNode = stateNode.GetActionNode(action)

			if childNode is None:
				break
			else:
				stateNode = childNode

		return stateNode




# region Rollouts
	def _RollOut(self, treeNode:TreeNode.TreeNode) -> SCT.Reward_List:

		valueFinalStates = False

		if self.Config["UseRealSim"]:
			isPlayingMask, totalRewards, states = self._RollOut_RealSim(treeNode.Env)
			valueFinalStates = self.Config["MonteCarloConfig"]["RollOutConfig"]["RealSim"]

		else:
			isPlayingMask, totalRewards, states = self._RollOut_ForwardModel(treeNode.State)
			valueFinalStates = self.Config["MonteCarloConfig"]["RollOutConfig"]["ForwardModel"]


		if valueFinalStates and self._ValueModel.CanPredict():
			with self._Metrics.Time("FinalStateValue"):
				values = self._ValueModel.Predict(states)
				totalRewards += values * isPlayingMask

		return totalRewards

	def _RollOut_RealSim(self, env:BaseEnv.BaseEnv) -> typing.Tuple[SCT.Action_List, SCT.Reward_List, SCT.State_List]:
		rollOutConfig = self.Config["MonteCarloConfig"]["RollOutConfig"]["RealSim"]
		numRollOuts = rollOutConfig["MaxRollOutCount"]
		timeOutAllowed = rollOutConfig["TimeOutAllowed"]
		maxDepth = rollOutConfig["MaxRollOutDepth"]


		with self._Metrics.Time("CloneState"):
			envs = [env.Clone() for _ in range(numRollOuts)]

		isPlayingMask = np.ones(numRollOuts, dtype=np.bool_)
		totalRewards = np.zeros(numRollOuts, dtype=np.float32)
		states = [None] * numRollOuts

		for d in range(maxDepth):

			# get the actions
			actions = MonteCarloAgent._SampleActions(self.ActionSpace, numRollOuts)

			for i in range(len(envs)):

				if envs[i].IsDone():
					continue

				nextState, reward, _, _, _ = envs[i].Step(actions[i])
				totalRewards[i] += reward
				states[i] = nextState

			if timeOutAllowed and time.process_time() < self._StopTime:
				break

		isPlayingMask = np.array([not env.IsDone() for env in envs], dtype=np.bool_)

		return isPlayingMask, totalRewards, states

	def _RollOut_ForwardModel(self, state:SCT.State):
		rollOutConfig = self.Config["MonteCarloConfig"]["RollOutConfig"]["ForwardModel"]
		numRollOuts = rollOutConfig["MaxRollOutCount"]
		timeOutAllowed = rollOutConfig["TimeOutAllowed"]
		maxDepth = rollOutConfig["MaxRollOutDepth"]

		with self._Metrics.Time("CloneState"):
			states, envs = TreeNode.TreeNode.CloneState(state, None, numRollOuts, False)

		isPlayingMask = np.ones(numRollOuts, dtype=np.bool_)
		totalRewards = np.zeros(numRollOuts, dtype=np.float32)

		for d in range(maxDepth):

			# get the actions
			actions = MonteCarloAgent._SampleActions(self.ActionSpace, numRollOuts)

			# predict the next states and rewards
			nextStates, envs, rewards, terminateds = self._ForwardModel.Predict(states, envs, actions)


			totalRewards += rewards * isPlayingMask
			isPlayingMask = np.logical_or(isPlayingMask, terminateds)

			# update the states
			states = nextStates

			if timeOutAllowed and time.process_time() < self._StopTime:
				break

		return isPlayingMask, totalRewards, states
# endregion
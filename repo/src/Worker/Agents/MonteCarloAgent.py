import Agents.BaseAgent as BaseAgent
import numpy as np
import Utils.SharedCoreTypes as SCT
from numpy.typing import NDArray
import time
from gymnasium.spaces import Discrete, Box
import Agents.ForwardModel as ForwardModel
import typing
from typing import Optional
import math


class TreeNode:
	def __init__(self,
			state:SCT.State,
			done:bool,
			parent:Optional['TreeNode'] = None,
			actionIdxTaken:Optional[int] = None):

		self.State = state
		self.Done = done

		self.TotalRewards:SCT.Reward = 0
		self.Counts = 0

		self.Parent = parent
		self.ActionIdxTaken = actionIdxTaken

		self.Children:Optional[typing.List['TreeNode']] = None

		return

	def Selection(self, exploreFactor:float) -> 'TreeNode':

		if self.Done:
			return self

		if self.Children is None:
			return self


		nodeScores = np.array([child.GetNodeScore(exploreFactor, self.Counts) for child in self.Children])

		selectedIndex = np.argmax(nodeScores)
		selectedNode = self.Children[selectedIndex]

		return selectedNode.Selection(exploreFactor)

	def Expand(self, actionList:SCT.Action_List, forwardModel:ForwardModel.ForwardModel) -> None:

		stateList = MonteCarloAgent.CloneState(self.State, len(actionList))

		nextStates, rewards, terminateds, truncateds = forwardModel.Predict(stateList, actionList)

		self.Children = []
		for i in range(len(nextStates)):
			done = terminateds[i] or truncateds[i]
			expandedNode = TreeNode(
				nextStates[i],
				done=done,
				parent=self,
				actionIdxTaken=i)


			if done:
				expandedNode.BackPropagate(rewards[i], counts=1)

			self.Children.append(expandedNode)

		return

	def BackPropagate(self, totalReward:SCT.Reward, counts:int) -> None:
		self.TotalRewards += totalReward
		self.Counts += counts

		if self.Parent is not None:
			self.Parent.BackPropagate(totalReward, counts)

		return

	def GetNodeScore(self, exploreFactor:float, perentCounts:int) -> float:
		# Unexplored nodes have maximum values so we favour exploration
		if self.Counts == 0:
			return float('inf')

		if self.Done:
			return float('-inf')


		parentCounts = self.Counts
		if self.Parent is not None:
			parentCounts = self.Parent.Counts

		avgReward = self.TotalRewards / self.Counts

		exploreValue = math.sqrt(math.log(parentCounts) / self.Counts)
		return avgReward + exploreFactor * exploreValue

	def GetActionValues(self) -> Optional[NDArray[np.float32]]:

		if self.Children is None:
			return None

		actionValues = np.zeros(len(self.Children), dtype=np.float32)
		for i in range(len(self.Children)):
			child = self.Children[i]
			if child.Counts > 0:
				actionValues[i] = child.TotalRewards / child.Counts

		return actionValues

	def GetActionNode(self, action:SCT.Action) -> 'TreeNode':
		for child in self.Children:
			if child.ActionTaken == action:
				return child

		return None

	def DetachParent(self) -> None:
		del self.Parent
		self.Parent = None
		self.ActionIdxTaken = None
		return



class MonteCarloAgent(BaseAgent.BaseAgent):

	def __init__(self, overrideConfig:SCT.Config, mode:BaseAgent.AgentMode, forwardModel:ForwardModel.ForwardModel):
		super().__init__(overrideConfig, mode=mode)

		subAgentConfig = self.Config.get("SubAgentConfig", {})
		self._SubAgent = BaseAgent.GetAgent(self.Config["SubAgent"], subAgentConfig, mode, forwardModel)
		self._ForwardModel = forwardModel
		self._CachedTree = None
		self._StopTime = 0
		return

	def Save(self, path:str) -> None:
		super().Save(path)
		self._SubAgent.Save(path)
		return

	def Load(self, path:str) -> None:
		super().Load(path)
		self._SubAgent.Load(path)
		return

	def GetAction(self, state:SCT.State) -> SCT.Action:
		super().GetAction(state)
		actionValues = self.GetActionValues(state)
		return self._GetMaxValues(actionValues)

	def GetActionValues(self, state:SCT.State) -> NDArray[np.float32]:
		super().GetActionValues(state)

		if not self._ForwardModel.CanPredict():
			return self._SubAgent.GetActionValues(state)

		self._StopTime = time.process_time() + self.Config["MaxSecondsPerAction"]

		rootNode = self._CachedTree
		if rootNode is None or rootNode.State != state:
			self._CachedTree = None
			rootNode = TreeNode(state, done=False)

		if rootNode.Children is None:
			rootNode.Expand(self.DataManager.ActionList, self._ForwardModel)

		# monte carlo tree search
		for i in range(self.Config["MaxSelections"]):
			# 1. selection
			selectedNode = rootNode.Selection(self.Config["ExploreFactor"])

			# 2. expansion
			if selectedNode.Counts > 0 and selectedNode.Children is None:
				selectedNode.Expand(self.DataManager.ActionList, self._ForwardModel)
				selectedNode = selectedNode.Selection(self.Config["ExploreFactor"])

			# 3. simulation
			totalRewards = self._RollOut(selectedNode.State)

			# 4. backpropagation
			selectedNode.BackPropagate(totalRewards.sum(), len(totalRewards))

			if time.process_time() >= self._StopTime:
				break

		if self.Config["CacheTree"]:
			self._CachedTree = rootNode

		if rootNode.Children is None:
			return self._SubAgent.GetActionValues(state)


		actionValues = rootNode.GetActionValues()
		return actionValues

	def Reset(self) -> None:
		super().Reset()
		self._SubAgent.Reset()
		return

	def Remember(self,
			state:SCT.State,
			action:SCT.Action,
			reward:SCT.Reward,
			nextState:SCT.State,
			terminated:bool,
			truncated:bool) -> None:

		super().Remember(state, action, reward, nextState, terminated, truncated)
		self._SubAgent.Remember(state, action, reward, nextState, terminated, truncated)

		# trim the tree
		if self._CachedTree is not None:

			# check if the root is the same state
			if self._CachedTree.State != state:
				self._CachedTree = None
			else:
				self._CachedTree = self._CachedTree.GetActionNode(action)
				self._CachedTree.DetachParent()
		return


	def _RollOut(self, state:SCT.State) -> SCT.Reward_List:

		numRollOuts = self.Config["RollOutConfig"]["MaxRollOutCount"]
		maxDepth = self.Config["RollOutConfig"]["MaxRollOutDepth"]
		timeOutAllowed = self.Config["RollOutConfig"]["TimeOutAllowed"]


		states = MonteCarloAgent.CloneState(state, numRollOuts)

		isPlayingMask = np.ones(numRollOuts, dtype=np.bool_)
		totalRewards = np.zeros(numRollOuts, dtype=np.float32)

		for d in range(maxDepth):

			# get the actions
			actions = self._SampleActions(self.DataManager.ActionSpace, numRollOuts)

			# predict the next states and rewards
			nextStates, rewards, terminateds, truncateds = self._ForwardModel.Predict(states, actions)


			totalRewards += rewards * isPlayingMask
			isPlayingMask = np.logical_or(isPlayingMask, terminateds, truncateds)

			# update the states
			states = nextStates

			if timeOutAllowed and time.process_time() < self._StopTime:
				break

		# add the predicted rewards to the total rewards if the game is still playing
		if self.Config["RollOutConfig"]["ValueFinalStates"] and np.any(isPlayingMask):

			for i in range(len(isPlayingMask)):
				if not isPlayingMask[i]:
					continue

				stateValue = self._SubAgent.GetActionValues(states[i]).max(axis=0)

				totalRewards[i] += stateValue


		return totalRewards

	@staticmethod
	def CloneState(state:SCT.State, count:int) -> SCT.State_List:

		if isinstance(state, int):
			return np.full(count, state, dtype=np.int_)


		currentState = state.reshape(1, -1)
		states = np.repeat(currentState, count, axis=0)
		return states

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


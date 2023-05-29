import src.Agents.BaseAgent as BaseAgent
import numpy as np
import src.Utils.SharedCoreTypes as SCT
from numpy.typing import NDArray
import time
from gymnasium.spaces import Discrete, Box
import src.Agents.ForwardModel as ForwardModel
import typing
import math




class TreeNode:
	def __init__(self,
			state:SCT.State,
			done:bool,
			parent:'TreeNode'=None,
			actionIdxTaken=None):
		self.State = state
		self.Done = done

		self.TotalRewards = 0
		self.Counts = 0

		self.Parent = parent
		self.ActionIdxTaken = actionIdxTaken

		self.Children = None

		return

	def Selection(self, exploreFactor:float) -> typing.Tuple['TreeNode', SCT.Action]:



		nodeScores = np.array([child.GetNodeScore(exploreFactor, self.Counts) for child in self.Children])

		selectedIndex = np.argmax(nodeScores)
		selectedNode = self.Children[selectedIndex]

		return selectedNode

	def Expand(self) -> None:


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


		parentCounts = self.Counts
		if self.Parent is not None:
			parentCounts = self.Parent.Counts

		avgReward = self.TotalRewards / self.Counts

		exploreValue = math.sqrt(math.log(parentCounts) / self.Counts)
		return avgReward + exploreFactor * exploreValue

	def GetActionNode(self, action:SCT.Action) -> 'TreeNode':
		for child in self.Children:
			if child.ActionTaken == action:
				return child

		return None

	def DetachParent(self):
		del self.Parent
		self.Parent = None
		self.ActionIdxTaken = None
		return



class MonteCarloAgent(BaseAgent.BaseAgent):

	def __init__(self, envConfig:SCT.Config, mode:BaseAgent.AgentMode, forwardModel:ForwardModel.ForwardModel):
		super().__init__(envConfig, mode=mode)

		self._SubAgent = BaseAgent.GetAgent(self.Config["SubAgent"], self.Config, mode, forwardModel)
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

		self._StopTime = time.process_time() + self.Config["MaxSecondsPerAction"]

		rootNode = self._CachedTree
		if rootNode is None or rootNode.State != state:
			self._CachedTree = None
			rootNode = TreeNode(state, self._SubAgent.GetActionValues(state), done=False)

		# monte carlo tree search
		for i in range(self.Config["MaxExpansions"]):
			# 1. selection
			selectedNode, selectedAction = rootNode.Selection()

			# 2. expansion
			nextState, reward, terminated, truncated = self._ForwardModel.Predict(selectedNode.State, selectedAction)

			expandedNode = TreeNode(
				nextState,
				self._SubAgent.GetActionValues(state),
				done=terminated or truncated,
				parent=selectedNode,
				actionTaken=selectedAction)


			if expandedNode.Done:
				expandedNode.BackPropagate(reward, counts=1)
				continue


			totalRewards = self._RollOut(expandedNode.State)
			avgReward = totalRewards.mean()


			expandedNode.BackPropagate(avgReward)

			if time.process_time() >= self._StopTime:
				break

		if self.Config["CacheTree"]:
			self._CachedTree = rootNode

		return rootNode

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


		# copy the state into a new array of states
		currentState = state.reshape(1, -1)
		states = np.repeat(currentState, numRollOuts, axis=0)

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


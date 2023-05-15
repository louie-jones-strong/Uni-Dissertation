import os
import typing

import numpy as np
import src.Agents.BaseAgent as BaseAgent
import src.DataManager.DataColumnTypes as DCT
import src.Utils.SharedCoreTypes as SCT
import tensorflow as tf
from numpy.typing import NDArray


class DQNAgent(BaseAgent.BaseAgent):

	def __init__(self,
			observationSpace:SCT.StateSpace,
			actionSpace:SCT.ActionSpace,
			envConfig:SCT.Config,
			mode:BaseAgent.AgentMode = BaseAgent.AgentMode.Train):
		super().__init__(observationSpace, actionSpace, envConfig, mode=mode)

		self.PriorityKey = "DQNAgent"

		self.RunModel = self.BuildModel()
		self.RunModel.summary()

		self.ExplorationRate = self.Config["MaxExplorationRate"]



		self.TrainingModel = self.BuildModel()
		self.TrainingModel.set_weights(self.RunModel.get_weights())

		constructor = BaseAgent.GetAgent(self.Config["ExplorationAgent"])
		self.ExplorationAgent = constructor(self._ObservationSpace, self._ActionSpace, envConfig)
		return

	def BuildModel(self) -> tf.keras.Model:
		inputShape = SCT.JoinTuples(self._ObservationSpace.shape, None)
		outputNumber = self._ActionSpace.n


		model = tf.keras.models.Sequential()

		# input layer
		model.add(tf.keras.layers.Input(shape=inputShape))

		# hidden layers
		if len(inputShape) > 1:
			model.add(tf.keras.layers.Conv2D(32, 8, strides=4, activation="relu"))
			model.add(tf.keras.layers.Conv2D(64, 4, strides=2, activation="relu"))
			model.add(tf.keras.layers.Conv2D(64, 3, strides=1, activation="relu"))

		model.add(tf.keras.layers.Flatten())

		# check if we are using dueling network
		if self.Config["DuellingNetwork"]:

			# value stream (value of the state)
			valueStream = tf.keras.models.Sequential()
			valueStream.add(tf.keras.layers.Dense(512, activation="relu"))
			valueStream.add(tf.keras.layers.Dense(1))

			# advantage stream (advantage of each action)
			advantageStream = tf.keras.models.Sequential()
			advantageStream.add(tf.keras.layers.Dense(512, activation="relu"))
			advantageStream.add(tf.keras.layers.Dense(outputNumber))


			# combine the two streams
			model.add(tf.keras.layers.Lambda(
				lambda x: tf.expand_dims(x[:, 0], -1) + x[:, 1:] - tf.reduce_mean(x[:, 1:], axis=1, keepdims=True)))

			model.add(valueStream)
			model.add(advantageStream)

			# output layer
			model.add(tf.keras.layers.Dense(outputNumber))

		else:
			model.add(tf.keras.layers.Dense(512, activation="relu"))

			# output layer
			model.add(tf.keras.layers.Dense(outputNumber))

		lr = self.Config["LearningRate"]
		optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

		# compile the network
		self.LossFunc = tf.keras.losses.Huber()
		model.compile(optimizer=optimizer, loss=self.LossFunc)
		return model




	def Reset(self) -> None:
		super().Reset()

		if self.Mode == BaseAgent.AgentMode.Train:
			self.ExplorationAgent.Reset()

		return

	def Remember(self,
			state:SCT.State,
			action:SCT.Action,
			reward:SCT.Reward,
			nextState:SCT.State,
			terminated:bool,
			truncated:bool) -> None:

		super().Remember(state, action, reward, nextState, terminated, truncated)

		if self.Mode == BaseAgent.AgentMode.Train:
			self.ExplorationAgent.Remember(state, action, reward, nextState, terminated, truncated)

		framesPerTrain = self.Config["FramesPerTrain"]
		if self.TotalRememberedFrame % framesPerTrain == 0:
			self.Train()
		return

	def Train(self) -> None:

		# check that we are in training mode
		if self.Mode != BaseAgent.AgentMode.Train:
			return


		if len(self.DataManager._ReplayBuffer) < self.Config["MinBufferSize"]:
			return

		# samples from the replay buffer

		def GetSamples(bactchSize:int) -> typing.Tuple[
				NDArray[np.int_],
				SCT.State_List,
				SCT.Action_List,
				SCT.Reward_List,
				SCT.State_List,
				NDArray[np.bool_],
				SCT.Reward_List,
				NDArray[np.float32]]:
			columns = [
				DCT.DataColumnTypes.CurrentState,
				DCT.DataColumnTypes.NextState,
				DCT.DataColumnTypes.Action,
				DCT.DataColumnTypes.Reward,
				DCT.DataColumnTypes.MaxFutureRewards,
				DCT.DataColumnTypes.Terminated,
				DCT.DataColumnTypes.Truncated,
			]

			priorityScale = self.Config["PriorityScale"]

			columnsData = self.DataManager.GetColumns(columns)
			indexs, priorities, samples = self.DataManager.SampleArrays(columnsData, bactchSize, self.PriorityKey, priorityScale=priorityScale)

			states, nextStates, actions, rewards, futureRewards, terminateds, truncateds = samples
			dones = np.logical_or(terminateds, truncateds)
			dones = dones.astype(np.float32)

			return indexs, states, actions, rewards, nextStates, dones, futureRewards, priorities

		def CalTargetQs(
				states:SCT.State_List,
				actions:SCT.Action_List,
				rewards:SCT.Reward_List,
				nextStates:SCT.State_List,
				dones:NDArray[np.bool_],
				futureRewards:SCT.Reward_List) -> SCT.Reward_List:

			# nextStatePredictedQ = self.RunModel.predict(nextStates, verbose=0)
			# # get the max Q for the next state
			# futureQ = np.max(nextStatePredictedQ, axis=1)


			# # if the real future reward is not None, then check if it is better than the predicted
			# if futureRewards is not None and self.Config["PropagateFutureReward"]:
			# 	futureQ = np.maximum(futureQ, futureRewards)

			# # we should not add future reward if it is the last state
			# futureQ *= (1-dones)

			# # discount factor for future rewards
			# gamma = self.Config["FutureRewardDiscount"]
			# futureQ *= gamma


			# # Calculate targets (bellman equation)
			# targetQs:SCT.Reward_List = rewards + futureQ








			# training DQN estimates best action in new states
			arg_q_max = self.TrainingModel.predict(nextStates, verbose=0).argmax(axis=1)

			# Target DQN estimates q-vals for new states
			future_q_vals = self.RunModel.predict(nextStates, verbose=0)
			double_q = future_q_vals[range(len(dones)), arg_q_max]

			# Calculate targets (bellman equation)
			gamma = self.Config["FutureRewardDiscount"]
			target_q = rewards + (gamma*double_q * (1-dones))











			return target_q

		def TrainWeights(targetQs:SCT.Reward_List,
				states:SCT.State_List,
				actions:SCT.Action_List,
				importance:NDArray[np.float32]) -> typing.Tuple[NDArray[np.float32], NDArray[np.float32]]:

			# aplly gradient descent
			with tf.GradientTape() as tape:
				qValues = self.TrainingModel(states)

				actionMask = tf.keras.utils.to_categorical(actions, self._ActionSpace.n, dtype=np.float32)
				currentQ = tf.reduce_sum(tf.multiply(qValues, actionMask), axis=1)

				absError = abs(targetQs - currentQ)
				loss = self.LossFunc(targetQs, currentQ)
				loss = tf.reduce_mean(loss * importance)

			gradients = tape.gradient(loss, self.TrainingModel.trainable_variables)
			self.TrainingModel.optimizer.apply_gradients(zip(gradients, self.TrainingModel.trainable_variables))
			return loss, absError


		batchSize = self.Config["BatchSize"]
		indexs, states, actions, rewards, nextStates, dones, futureRewards, priorities = GetSamples(batchSize)
		importance = priorities ** (1-self.ExplorationRate)


		targetQs = CalTargetQs(states, actions, rewards, nextStates, dones, futureRewards)
		loss, absError = TrainWeights(targetQs, states, actions, importance)


		self._Logger.LogDict({"DQN_loss": loss.numpy()})


		assert np.all(absError >= 0), f"absError should be positive, but got {absError}"

		# update the priorities
		self.DataManager._ReplayBuffer.UpdatePriorities(self.PriorityKey, indexs, absError)


		# update the training network
		if self.TotalRememberedFrame % self.Config["FramesPerUpdateRunningNetwork"] == 0:
			self.RunModel.set_weights(self.TrainingModel.get_weights())
			print("=================update running network=================")
		return


	def GetAction(self, state:SCT.State) -> SCT.Action:
		super().GetAction(state)
		actionValues = self.GetActionValues(state)
		return self._GetMaxValues(actionValues)

	def GetActionValues(self, state:SCT.State) -> NDArray[np.float32]:

		isExploreAction = False
		# if it is training mode
		if self.Mode == BaseAgent.AgentMode.Train:
			if len(self.DataManager._ReplayBuffer) < self.Config["MinBufferSize"]:
				isExploreAction = True
			else:
				if np.random.random() < self.ExplorationRate:
					isExploreAction = True

				# decay exploration rate
				self.ExplorationRate -= self.Config["ExplorationDelta"]
				self.ExplorationRate = max(self.ExplorationRate, self.Config["MinExplorationRate"])




		# get action values
		if isExploreAction:
			# get action values from the exploration agent
			actionValues = self.ExplorationAgent.GetActionValues(state)
		else:
			# get action values from the network
			state = np.expand_dims(state, axis=0)
			actionValues = self.RunModel.predict(state, verbose=0)[0]
		return actionValues


	def Save(self, path:str) -> None:
		super().Save(path)
		self.RunModel.save(os.path.join(path, "DqnModel.h5"))
		return

	def Load(self, path:str) -> None:
		super().Load(path)

		modelPath = os.path.join(path, "DqnModel.h5")
		if os.path.exists(modelPath):
			self.RunModel = tf.keras.models.load_model(modelPath)

			if self.Mode == BaseAgent.AgentMode.Train:
				self.TrainingModel.set_weights(self.RunModel.get_weights())

		return
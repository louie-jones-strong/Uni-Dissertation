from . import BaseAgent
import tensorflow as tf
import numpy as np

class DQNAgent(BaseAgent.BaseAgent):

	def __init__(self, env, replayBuffer, mode=BaseAgent.AgentMode.Train):
		super().__init__(env, replayBuffer, mode=mode)


		self.LossFunc = tf.keras.losses.Huber()
		self.RunModel = self.BuildModel()

		if self.Mode == BaseAgent.AgentMode.Train:
			self.TrainingModel = self.BuildModel()

			self.ExplorationAgent = BaseAgent.GetAgent(self.Config["ExplorationAgent"])(self.Env, self.ReplayBuffer)
			self.ExplorationRate = 1.0
			self.IsEval = True

		return

	def BuildModel(self):
		inputShape = self.Env.observation_space.shape
		outputNumber = self.Env.action_space.n


		model = tf.keras.models.Sequential()

		# input layer
		model.add(tf.keras.layers.Input(shape=inputShape))
		model.add(tf.keras.layers.Flatten())

		# hidden layers


		# output layer
		model.add(tf.keras.layers.Dense(outputNumber))

		lr = self.Config["LearningRate"]
		optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

		# compile the network
		model.compile(optimizer=optimizer, loss=self.LossFunc)

		return model




	def Reset(self):
		super().Reset()
		self.ExplorationAgent.Reset()

		if self.IsEval:
			self.IsEval = False
		elif self.Mode == BaseAgent.AgentMode.Train:
			if self.EpisodeNum % self.Config["EpisodesBetweenEval"] == 0:
				self.IsEval = True

		return

	def Train(self):

		# check that we are in training mode
		if self.Mode != BaseAgent.AgentMode.Train:
			return

		if len(self.ReplayBuffer) < self.Config["MinBufferSize"]:
			return

		# samples from the replay buffer
		batchSize = self.Config["BatchSize"]
		states, actions, rewards, nextStates, dones = self.ReplayBuffer.Sample(batchSize)
		states = np.array(states)
		nextStates = np.array(nextStates)
		dones = tf.convert_to_tensor([float(done) for done in dones])



		def CalTargetQs(states, actions, rewards, nextStates, dones):

			# Run network predicted q-values for next states
			arg_q_max = self.RunModel.predict(nextStates, verbose=0).argmax(axis=1)

			# training network predicted q-values for next states
			future_q_vals = self.TrainingModel.predict(nextStates, verbose=0)

			double_q = future_q_vals[range(len(states)), arg_q_max]


			# Calculate targets (bellman equation)
			gamma = self.Config["FutureRewardDiscount"]
			targetQs = rewards + (gamma*double_q * (1-dones))

			return targetQs

		def TrainWeights(targetQs, states, actions):
			# aplly gradient descent
			with tf.GradientTape() as tape:
				qValues = self.TrainingModel(states)

				actionMask = tf.keras.utils.to_categorical(actions, self.Env.action_space.n, dtype=np.float32)
				currentQ = tf.reduce_sum(tf.multiply(qValues, actionMask), axis=1)

				loss = self.LossFunc(targetQs, currentQ)

			model_gradients = tape.gradient(loss, self.TrainingModel.trainable_variables)
			self.TrainingModel.optimizer.apply_gradients(zip(model_gradients, self.TrainingModel.trainable_variables))
			return loss


		targetQs = CalTargetQs(states, actions, rewards, nextStates, dones)
		loss = TrainWeights(targetQs, states, actions)


		# update the training network
		if self.TotalFrameNum % self.Config["FramesPerUpdateRunningNetwork"] == 0:
			self.RunModel.set_weights(self.TrainingModel.get_weights())

		return

	def GetActionValues(self, state):

		isExploreAction = False
		# if it is training mode
		if self.Mode == BaseAgent.AgentMode.Train and not self.IsEval:
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

		framesPerTrain = self.Config["FramesPerTrain"]
		if self.TotalFrameNum % framesPerTrain == 0:
			self.Train()


		return actionValues
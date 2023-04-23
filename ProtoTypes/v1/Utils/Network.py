from tensorflow.keras import models, layers, Input, regularizers, optimizers, callbacks

class Network:

	def __init__(self):
		self.Model = None
		return

	def BuildModel(self, inputShape, outputNumber,
					layer1Neurons=-1,
					layer2Neurons=-1,
					layer3Neurons=-1,
					dropout=-1, l1=-1, l2=-1,
					optimizer="rmsprop",
					lr=0.001):

		def TryAddLayer(network, layerNeurons, dropout, l1, l2):
			if layerNeurons > 1:

				# check if we need to add a L1 or L2
				if l1 > 0 or l2 > 0:
					network.add(layers.Dense(layerNeurons, activation='relu', kernel_regularizer=regularizers.L1L2(l1=l1, l2=l2)))
				else:
					network.add(layers.Dense(layerNeurons, activation='relu'))


				# check if we need to add a drop out layer
				if dropout > 0:
					network.add(layers.Dropout(dropout))


			return layerNeurons > 1

		network = models.Sequential()
		# input layer
		network.add(layers.Input(shape=inputShape))
		network.add(layers.Flatten())

		if TryAddLayer(network, layer1Neurons, dropout, l1, l2):

			# only add the second hidden layer if the first one was added
			if TryAddLayer(network, layer2Neurons, dropout, l1, l2):

				# only add the third hidden layer if the second one was added
				TryAddLayer(network, layer3Neurons, dropout, l1, l2)

		# output layer
		network.add(layers.Dense(outputNumber, activation='softmax'))

		# define the optimizer
		if optimizer == "rmsprop":
			optimizer = optimizers.RMSprop(learning_rate=lr)
		elif optimizer == "adam":
			optimizer = optimizers.Adam(learning_rate=lr)
		elif optimizer == "adadelta":
			optimizer = optimizers.Adadelta(learning_rate=lr)
		elif optimizer == "adamax":
			optimizer = optimizers.Adamax(learning_rate=lr)
		elif optimizer == "nadam":
			optimizer = optimizers.Nadam(learning_rate=lr)
		elif optimizer == "sgd":
			optimizer = optimizers.SGD(learning_rate=lr)
		else:
			raise Exception("Unknown optimizer: " + optimizer)

		# compile the network
		network.compile(
			optimizer=optimizer,
			loss='categorical_crossentropy',
			metrics=['accuracy'])

		self.Model = network
		return

	def Train(self, trainInputs, trainTargets,
				epochs, batchSize,
				validationData=None,
				callbackList=[], verbose=1):

		# train the network
		history = self.Model.fit(
			trainInputs, trainTargets,
			shuffle=True,
			epochs=epochs, batch_size=batchSize,
			validation_data=validationData,
			callbacks=callbackList,
			verbose=verbose)
		return history

	def Predict(self, inputs):
		return self.Model.predict(inputs, verbose=0)
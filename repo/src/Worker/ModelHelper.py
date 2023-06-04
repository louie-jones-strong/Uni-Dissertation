import tensorflow as tf



ModelTypes = ["policy", "value", "forward"]


def GetModel(modelType, observationSpace, actionSpace, config):

	model = FetchModel(modelType)
	if model is None:
		model = BuildModel(modelType, observationSpace, actionSpace, config)

	return model

def BuildModel(modeType, observationSpace, actionSpace, config):

	if modeType == "policy":
		model = _Build_Policy(observationSpace, actionSpace, config)
	# elif modeType == "value":



	return model

def FetchModel(modelType):

	return

def PushModel(modelType, model):

	return








#region Build Models

def _Build_Policy(observationSpace, actionSpace, config):
	model = tf.keras.models.Sequential()

	# input layer
	model.add(tf.keras.layers.Input(shape=observationSpace))

	model.add(tf.keras.layers.Flatten())

	model.add(tf.keras.layers.Dense(512, activation="relu"))

	# output layer
	model.add(tf.keras.layers.Dense(actionSpace))

	optimizer = tf.keras.optimizers.Adam()

	# compile the network
	model.compile(optimizer=optimizer, loss=tf.keras.losses.Huber())


	return model

#endregion
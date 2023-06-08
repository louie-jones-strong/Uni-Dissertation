import tensorflow as tf
import redis
import numpy as np


ModelTypes = ["policy", "value", "forward"]

RedisClient = redis.Redis(host='model-store', port=5002, db=0)


def BuildModel(modeType, observationSpace, actionSpace, config):

	if modeType == "policy":
		model = _Build_Policy(observationSpace, actionSpace, config)
	# elif modeType == "value":



	return model


def FetchNewestWeights(modelType, model):
	flatWeightBytes = RedisClient.get(modelType)
	if flatWeightBytes is None:
		return False

	flatWeights = np.frombuffer(flatWeightBytes, dtype=np.float32)

	currentWeights = model.get_weights()

	newWeights = []
	idx = 0
	for layer in currentWeights:
		shape = layer.shape
		count = np.prod(shape)
		layerWeights = flatWeights[idx:idx+count]
		layerWeights = layerWeights.reshape(shape)
		idx += count
		newWeights.append(layerWeights)

	model.set_weights(newWeights)

	return True

def PushModel(modelType, model):
	weights = model.get_weights()
	flatWeights = np.concatenate([w.flatten() for w in weights])

	flatWeightBytes = flatWeights.tobytes()
	RedisClient.set(modelType, flatWeightBytes)
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
import reverb
import Common.Utils.UserInputHelper as UI
import os
from Common.Utils.PathHelper import GetRootPath
import tensorflow as tf
import numpy as np
import ModelHelper

BatchSize = 32
ItetationsPerUpdate = 100





if __name__ == "__main__":
	parser = UI.ArgParser()

	envConfigFolder = os.path.join(GetRootPath(), "Config", "Envs")
	parser.AddOptionsOption("model", "The type of model to train", ModelHelper.ModelTypes, "ModelType")
	parser.AddFilePathOption("env", "path to env config", envConfigFolder, "env")
	args = parser.GetArgs()

	envConfigPath = args["env"]
	modelType = args["model"]

	print("="*20)
	print("envConfigPath:", envConfigPath)
	print("modelType:", modelType)
	print()

	print("build model")
	model = ModelHelper.BuildModel(modelType, (1,), (1), {})
	print("built model")

	print("fetching newest weights")
	didFetch = ModelHelper.FetchNewestWeights(modelType, model)
	print("fetched newest weights", didFetch)

	print("Connecting to experience store")
	dataset = reverb.TrajectoryDataset.from_table_signature(
		server_address=f'experience-store:{5001}',
		table='Trajectories',
		max_in_flight_samples_per_worker=10)

	print("Connected to experience store")

	batched_dataset = dataset.batch(BatchSize)


	print("Starting training")
	for batch in batched_dataset.take(ItetationsPerUpdate):
		x = batch.data["State"]
		y = batch.data["Action"]
		print(x.shape)
		model.fit(x, y, epochs=1)

	print("Finished training")

	print("Saving model")
	ModelHelper.PushModel(modelType, model)


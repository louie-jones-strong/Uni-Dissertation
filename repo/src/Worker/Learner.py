import reverb
import Common.Utils.UserInputHelper as UI
import os
from Common.Utils.PathHelper import GetRootPath
import tensorflow as tf
import numpy as np
import ModelHelper

BatchSize = 32
EpochsPerUpdate = 10





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

	model = ModelHelper.GetModel(modelType, (1,), (1), {})



	dataset = reverb.TrajectoryDataset.from_table_signature(
		server_address=f'experience-store:{5001}',
		table='Trajectories',
		max_in_flight_samples_per_worker=10)

	batched_dataset = dataset.batch(BatchSize)



	# for epoch in range(EpochsPerUpdate):
	# 	batched_dataset.take(1)

	model.fit(batched_dataset, epochs=1)


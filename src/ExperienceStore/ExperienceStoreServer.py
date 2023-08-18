import reverb
import tensorflow as tf
import time
import src.Common.Enums.eDataColumnTypes as DCT
import src.Common.Utils.SharedCoreTypes as SCT
import src.Common.Utils.ConfigHelper as ConfigHelper
import gymnasium as gym
import typing

class ExperienceStoreServer(ConfigHelper.ConfigurableClass):
	def __init__(self, envConfig:SCT.Config):
		self.LoadConfig()
		self.EnvConfig = envConfig

		self.ObservationSpace = ConfigHelper.ConfigToSpace(self.EnvConfig["ObservationSpace"])
		self.ActionSpace = ConfigHelper.ConfigToSpace(self.EnvConfig["ActionSpace"])


		self.Tables = []
		for table in self.Config["DataTables"]:

			tableName = table["TableName"]
			stepCount = table["StepCount"]
			minSize = table["MinCount"]
			maxSize = table["MaxCount"]
			priorityExponent = table["PriorityExponent"]
			columns = table["Columns"]
			self.AddTableTrajectory(tableName, stepCount, minSize, maxSize, priorityExponent, columns)

		return

	def AddTableTrajectory(self,
			tableName:str,
			stepCount:int,
			minSize:int,
			maxSize:int,
			priorityExponent:float,
			columns:typing.List[str]) -> None:

		state_Spec = self.SpaceToSpec(self.ObservationSpace, stepCount)
		action_Spec = self.SpaceToSpec(self.ActionSpace, stepCount)

		reward_Spec = tf.TensorSpec([stepCount], tf.double)
		endFlag_Spec = tf.TensorSpec([stepCount], tf.bool)


		signature = {
				DCT.eDataColumnTypes.CurrentState.name: state_Spec,
				DCT.eDataColumnTypes.NextState.name: state_Spec,
				DCT.eDataColumnTypes.Action.name: action_Spec,
				DCT.eDataColumnTypes.Reward.name: reward_Spec,
				DCT.eDataColumnTypes.MaxFutureRewards.name: reward_Spec,
				DCT.eDataColumnTypes.Terminated.name: endFlag_Spec,
				DCT.eDataColumnTypes.Truncated.name: endFlag_Spec
			}

		filteredSignature = {}
		for key, value in signature.items():
			if key in columns:
				filteredSignature[key] = value


		table = reverb.Table(
			name=tableName,
			sampler=reverb.selectors.Prioritized(priority_exponent=priorityExponent),
			remover=reverb.selectors.Fifo(),
			max_size=maxSize,
			rate_limiter=reverb.rate_limiters.MinSize(minSize),
			signature=filteredSignature)

		self.Tables.append(table)
		return

	def SpaceToSpec(self, space:gym.Space, stepCount:int) -> tf.TensorSpec:

		spec = None

		if isinstance(space, gym.spaces.Discrete):
			spec = tf.TensorSpec((stepCount), tf.int64)

		elif isinstance(space, gym.spaces.Box):
			shape = SCT.JoinTuples((stepCount,), space.shape)

			spec = tf.TensorSpec(shape, tf.double)

		else:
			raise NotImplementedError()

		return spec

	def Run(self) -> None:

		_ = reverb.Server(tables=self.Tables,port=5001)

		print("server started")

		while True:
			time.sleep(1)

		return
import reverb
import tensorflow as tf
import time
import src.Common.Enums.eDataColumnTypes as DCT
import src.Common.Utils.SharedCoreTypes as SCT
import src.Common.Utils.ConfigHelper as ConfigHelper
import gymnasium as gym

def Run() -> None:

	# todo make this configurable
	TragetoryStepCount = 1

	MinSize = 1_000
	MaxSize = 1_000_000
	PriorityExponent = 0.8


	State_Spec = tf.TensorSpec([TragetoryStepCount], tf.int64)
	Action_Spec = tf.TensorSpec([TragetoryStepCount], tf.int64)

	Reward_Spec = tf.TensorSpec([TragetoryStepCount], tf.double)

	EndFlag_Spec = tf.TensorSpec([TragetoryStepCount], tf.bool)


	_ = reverb.Server(
		tables=[
			reverb.Table(
				name='Trajectories',
				sampler=reverb.selectors.Prioritized(priority_exponent=PriorityExponent),
				remover=reverb.selectors.Fifo(),
				max_size=MaxSize,
				rate_limiter=reverb.rate_limiters.MinSize(MinSize),
				signature={
					DCT.eDataColumnTypes.CurrentState.name: State_Spec,
					DCT.eDataColumnTypes.NextState.name: State_Spec,
					DCT.eDataColumnTypes.Action.name: Action_Spec,
					DCT.eDataColumnTypes.Reward.name: Reward_Spec,
					DCT.eDataColumnTypes.MaxFutureRewards.name: Reward_Spec,
					DCT.eDataColumnTypes.Terminated.name: EndFlag_Spec,
					DCT.eDataColumnTypes.Truncated.name: EndFlag_Spec
				},
			)
		],
		port=5001)

	print("server started")

	while True:
		time.sleep(1)

	return


class ExperienceStoreServer:
	def __init__(self, config:SCT.Config):

		# todo make this configurable
		self.ObservationSpace = ConfigHelper.ConfigToSpace(config["ObservationSpace"])
		self.ActionSpace = ConfigHelper.ConfigToSpace(config["ActionSpace"])


		self.Tables = []
		self.AddTableTrajectory("Forward_Trajectories", 1, 1_000, 1_000_000, 0.8)
		self.AddTableTrajectory("Value_Trajectories", 1, 1_000, 1_000_000, 0.8)


		return

	def AddTableTrajectory(self, tableName:str, stepCount:int, minSize:int, maxSize:int, priorityExponent:float) -> None:
		state_Spec = self.SpaceToSpec(self.ObservationSpace, stepCount)
		action_Spec = self.SpaceToSpec(self.ActionSpace, stepCount)

		reward_Spec = tf.TensorSpec([stepCount], tf.double)
		endFlag_Spec = tf.TensorSpec([stepCount], tf.bool)


		table = reverb.Table(
			name=tableName,
			sampler=reverb.selectors.Prioritized(priority_exponent=priorityExponent),
			remover=reverb.selectors.Fifo(),
			max_size=maxSize,
			rate_limiter=reverb.rate_limiters.MinSize(minSize),
			signature={
				DCT.eDataColumnTypes.CurrentState.name: state_Spec,
				DCT.eDataColumnTypes.NextState.name: state_Spec,
				DCT.eDataColumnTypes.Action.name: action_Spec,
				DCT.eDataColumnTypes.Reward.name: reward_Spec,
				DCT.eDataColumnTypes.MaxFutureRewards.name: reward_Spec,
				DCT.eDataColumnTypes.Terminated.name: endFlag_Spec,
				DCT.eDataColumnTypes.Truncated.name: endFlag_Spec
			})

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
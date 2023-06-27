import reverb
import tensorflow as tf
import time
import src.Common.Enums.DataColumnTypes as DCT

def Run() -> None:

	# todo make this configurable
	TragetoryStepCount = 1
	MaxSize = 1_000_000

	State_Spec = tf.TensorSpec([TragetoryStepCount], tf.int64)
	Action_Spec = tf.TensorSpec([TragetoryStepCount], tf.int64)

	Reward_Spec = tf.TensorSpec([TragetoryStepCount], tf.double)

	EndFlag_Spec = tf.TensorSpec([TragetoryStepCount], tf.bool)


	_ = reverb.Server(
		tables=[
			reverb.Table(
				name='Trajectories',
				sampler=reverb.selectors.Prioritized(priority_exponent=0.8),
				remover=reverb.selectors.Fifo(),
				max_size=MaxSize,
				# Sets Rate Limiter to a low number for the examples.
				# Read the Rate Limiters section for usage info.
				rate_limiter=reverb.rate_limiters.MinSize(2),
				signature={
					DCT.DataColumnTypes.CurrentState.name: State_Spec,
					DCT.DataColumnTypes.NextState.name: State_Spec,
					DCT.DataColumnTypes.Action.name: Action_Spec,
					DCT.DataColumnTypes.Reward.name: Reward_Spec,
					DCT.DataColumnTypes.MaxFutureRewards.name: Reward_Spec,
					DCT.DataColumnTypes.Terminated.name: EndFlag_Spec,
					DCT.DataColumnTypes.Truncated.name: EndFlag_Spec
				},
			)
		],
		port=5001)

	print("server started")

	while True:
		time.sleep(1)

	return
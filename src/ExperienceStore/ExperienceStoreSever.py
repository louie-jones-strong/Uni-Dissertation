import reverb
import tensorflow as tf
import time
import src.Common.Enums.eDataColumnTypes as DCT

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
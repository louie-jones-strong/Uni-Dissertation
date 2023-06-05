import reverb
import tensorflow as tf
import time


TragetoryStepCount = 1
MaxSize = 1_000_000

State_Spec = tf.TensorSpec([TragetoryStepCount], tf.int64)
Action_Spec = tf.TensorSpec([TragetoryStepCount], tf.int64)

Reward_Spec = tf.TensorSpec([TragetoryStepCount], tf.double)

EndFlag_Spec = tf.TensorSpec([TragetoryStepCount], tf.bool)

server = reverb.Server(
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
				"State": State_Spec,
				"NextState": State_Spec,
				"Action": Action_Spec,
				"Reward": Reward_Spec,
				"FutureReward": Reward_Spec,
				"Terminated": EndFlag_Spec,
				"Truncated": EndFlag_Spec
			},
		)
	],
	port=5001)

print("server started")

while True:
	time.sleep(1)
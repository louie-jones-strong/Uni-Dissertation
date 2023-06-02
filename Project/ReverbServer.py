import reverb
import tensorflow as tf


OBSERVATION_SPEC = tf.TensorSpec([10, 10], tf.uint8)
ACTION_SPEC = tf.TensorSpec([2], tf.float32)


simple_server = reverb.Server(
	tables=[
		reverb.Table(
			name='my_table',
			sampler=reverb.selectors.Prioritized(priority_exponent=0.8),
			remover=reverb.selectors.Fifo(),
			max_size=int(1e6),
			# Sets Rate Limiter to a low number for the examples.
			# Read the Rate Limiters section for usage info.
			rate_limiter=reverb.rate_limiters.MinSize(2),
			# The signature is optional but it is good practice to set it as it
			# enables data validation and easier dataset construction. Note that
			# we prefix all shapes with a 3 as the trajectories we'll be writing
			# consist of 3 timesteps.
			signature={
				'actions':
					tf.TensorSpec([3, *ACTION_SPEC.shape], ACTION_SPEC.dtype),
				'observations':
					tf.TensorSpec([3, *OBSERVATION_SPEC.shape],
								OBSERVATION_SPEC.dtype),
			},
		)
	],
	# Sets the port to None to make the server pick one automatically.
	# This can be omitted as it's the default.
	port=None)
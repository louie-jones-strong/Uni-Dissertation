import reverb


simple_server = reverb.Server(
	tables=[
		reverb.Table(
			name='Trajectories',
			sampler=reverb.selectors.Prioritized(priority_exponent=0.8),
			remover=reverb.selectors.Fifo(),
			max_size=int(1e6),
			# Sets Rate Limiter to a low number for the examples.
			# Read the Rate Limiters section for usage info.
			rate_limiter=reverb.rate_limiters.MinSize(2),
		)
	],
	port=8000)

while True:
	pass
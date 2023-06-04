import reverb

print()
print("Connecting to Reverb Server...")
client = reverb.Client(f'experience-store:{5001}')
print(client.server_info())
print()
print("Connected to Reverb Server!")

# # Creates a single item and data element [0, 1].
# client.insert([0, 1], priorities={'Trajectories': 1.0})


# print(list(client.sample('Trajectories', num_samples=2)))


# with client.trajectory_writer(num_keep_alive_refs=3) as writer:
# 	writer.append({'a': 2, 'b': 12})
# 	writer.append({'a': 3, 'b': 13})
# 	writer.append({'a': 4, 'b': 14})

# 	# Create an item referencing all the data.
# 	writer.create_item(
# 		table='Trajectories',
# 		priority=1.0,
# 		trajectory={
# 			'a': writer.history['a'][:],
# 			'b': writer.history['b'][:],
# 		})

# 	# Block until the item has been inserted and confirmed by the server.
# 	writer.flush()


print("creating Dataset...")

# Dataset samples sequences of length 3 and streams the timesteps one by one.
# This allows streaming large sequences that do not necessarily fit in memory.
dataset = reverb.TrajectoryDataset.from_table_signature(
	server_address=f'experience-store:{5001}',
	table='Trajectories',
	max_in_flight_samples_per_worker=10)

print("Dataset created!")


samples = 0
batched_dataset = dataset.batch(1)

print()
print(batched_dataset)
print()

for sample in batched_dataset.take(1):
	print(samples, sample.info.key)
	samples += 1

print("Consumed", samples, "samples!")


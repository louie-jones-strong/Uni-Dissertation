import reverb

print()
print("Connecting to Reverb Server...")
client = reverb.Client(f'experience-store:{5001}')
print(client.server_info())
print()
print("Connected to Reverb Server!")

# Creates a single item and data element [0, 1].
client.insert([0, 1], priorities={'my_table': 1.0})


print(list(client.sample('my_table', num_samples=2)))


with client.trajectory_writer(num_keep_alive_refs=3) as writer:
	writer.append({'a': 2, 'b': 12})
	writer.append({'a': 3, 'b': 13})
	writer.append({'a': 4, 'b': 14})

	# Create an item referencing all the data.
	writer.create_item(
		table='my_table',
		priority=1.0,
		trajectory={
			'a': writer.history['a'][:],
			'b': writer.history['b'][:],
		})

	# Block until the item has been inserted and confirmed by the server.
	writer.flush()
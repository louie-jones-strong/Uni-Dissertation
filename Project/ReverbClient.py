import reverb

print()
print("Connecting to Reverb Server...")
client = reverb.Client(f'reverb-server:{8000}')
print(client.server_info())
print()
print("Connected to Reverb Server!")

# Creates a single item and data element [0, 1].
client.insert([0, 1], priorities={'my_table': 1.0})


print(list(client.sample('my_table', num_samples=2)))
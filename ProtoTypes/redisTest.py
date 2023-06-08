import redis

# Connect to Redis
r = redis.Redis(host='model-store', port=6000, db=0)

# Push data to Redis
r.set('key', 'value')

# Fetch data from Redis
value = r.get('key')
print(value.decode())  # Convert bytes to string if needed

# Example with hash data structure
r.hset('hash_key', 'field1', 'value1')
r.hset('hash_key', 'field2', 'value2')

hash_value = r.hget('hash_key', 'field1')
print(hash_value.decode())  # Convert bytes to string if needed

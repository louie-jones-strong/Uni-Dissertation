import os


# create secrets.env file if it doesn't exist
if not os.path.exists("secrets.env"):
	with open("secrets.env", "w") as f:
		f.write("WANDB_API_KEY=")
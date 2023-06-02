import cv2
import numpy as np
import time

startTime = time.process_time_ns()

# Generate numpy array with gray frames
frames = []
for _ in range(20):
	frame = np.full((480, 640, 3), 128, dtype=np.uint8)
	frames.append(frame)

# Save frames as video file
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_file = 'output.mp4'
video_writer = cv2.VideoWriter(output_file, fourcc, 25, (640, 480))
for frame in frames:
	video_writer.write(frame)
video_writer.release()
print("Video file generated successfully.")
print("Time taken:", (time.process_time_ns() - startTime)/1_000_000_000, "seconds")

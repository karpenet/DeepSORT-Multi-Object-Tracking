import glob
import re
from tqdm import tqdm
from deepsort import DeepSort
import cv2
import time
import speedscope

# Initialize the DeepSort tracker
tracking = DeepSort()

# Get sorted list of image files
images_files = sorted(
    glob.glob("dataset/KITTI/image_03/data/*.png"),
    key=lambda x: float(re.findall("(\d+)", x)[0]),
)
images = []

# Load a subset of images
index = 57
for img in images_files[index : index + 2]:
    images.append(cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB))

# Process video images
video_images = images_files
result_video = []

# Measure FPS and overlay on each frame
prev_time = time.time()
for img in tqdm(video_images):
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    with speedscope.track('speedscope.json'):
        out_img, stored_obstacles = tracking.inference(
            cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
        )
    tracking.logger.log(f"Current FPS: {fps:.2f}")
    
    if out_img is not None:
    #     # Overlay FPS on the frame
    #     cv2.putText(out_img, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 10, (255, 0, 0), 2, cv2.LINE_AA)
        result_video.append(out_img)

#### Profiler Code ####
# Load a subset of images
# index = 57
# for img in images_files[index:index + 10]:
#     images.append(cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB))
# benchmark_inference(tracking.inference, images)
# image = cv2.cvtColor(cv2.imread(images_files[0]), cv2.COLOR_BGR2RGB)

# with speedscope.track('speedscope.json'):
#     tracking.inference(image)

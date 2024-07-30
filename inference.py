import glob
import re
from tqdm import tqdm
from deepsort import DeepSort
import cv2

# Initialize the DeepSort tracker
tracking = DeepSort()

# Get sorted list of image files
images_files = sorted(glob.glob("dataset/KITTI/image_03/data/*.png"), key=lambda x: float(re.findall("(\d+)", x)[0]))
images = []

# Load a subset of images
index = 57
for img in images_files[index:index+2]:
    images.append(cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB))

# Process video images
video_images = images_files
result_video = []

for img in tqdm(video_images):
    out_img, stored_obstacles = tracking.inference(cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB))
    if out_img is not None:
        result_video.append(out_img)

# Save the result video
out = cv2.VideoWriter('output/output_deepsort.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 15, (out_img.shape[1], out_img.shape[0]))

for img in result_video:
    out.write(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
out.release()
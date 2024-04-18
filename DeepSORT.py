# %% [markdown]
# # DeepSORT using YOLO

# %% [markdown]
# ## Imports

# %%
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import copy
import glob
import pickle
import cv2
import re
import numpy as np

import torch
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils
from torch.utils.data import DataLoader,Dataset
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import yolov5 # pip3 install yolo5

import glob 
from tqdm import tqdm

# %% [markdown]
# ## Load & Visualize the Images

# %%
def visualize_images(input_images):
    fig=plt.figure(figsize=(150,75))
    for i in range(len(input_images)):
        fig.add_subplot(1, len(input_images), i+1)
        plt.imshow(input_images[i])
    plt.show()

# %%
#Indexes to test on:
# 100 — Same number of detections
# 300 — 2 Lost Tracks
# 782 — 1 New Detection

images_files = sorted(glob.glob("images/*.jpg"), key=lambda x:float(re.findall("(\d+)",x)[0]))
images = []

index = 57
for img in images_files[index:index+2]:
    images.append(cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB))

# %%
visualize_images(images)

# %% [markdown]
# ## 1) YOLOv5 — Object Detection

# %%
model = yolov5.load('yolov5s.pt')

# %%
model.conf = 0.5
model.iou = 0.4

# %%
classesFile = "coco.names"
with open(classesFile,'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

def id_to_color(idx):
    """
    Random function to convert an id to a color
    """
    blue = idx*5 % 256
    green = idx*12 %256
    red = idx*23 %256
    return (red, green, blue)

def draw_boxes_v5(image, boxes, categories, mot_mode=False):
    h, w, _ = image.shape
    for i, box in enumerate(boxes):
        label = classes[int(categories[i])]
        color = id_to_color(i*10) if mot_mode==True else (255,0,0)
        cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, thickness=7)
        cv2.putText(image, str(label), (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), thickness=3)
    return image

def inference(img):
    results = model(img)
    predictions = results.pred[0]
    boxes = predictions[:, :4].tolist()
    boxes_int = [[int(v) for v in box] for box in boxes]
    scores = predictions[:,4].tolist()
    categories = predictions[:,5].tolist()
    categories_int = [int(c) for c in categories]
    img_out = draw_boxes_v5(img, boxes_int, categories_int, mot_mode=True)
    return img_out, boxes_int, categories_int, scores

# %%
yolo_images = []
yolo_boxes = []
yolo_categories = []
yolo_scores = []
pics = copy.deepcopy(images)

for img in pics:
    result, pred_bboxes, pred_categories, pred_scores = inference(img)
    yolo_boxes.append(pred_bboxes)
    yolo_images.append(result)
    yolo_categories.append(pred_categories)
    yolo_scores.append(pred_scores)

# %%
visualize_images(yolo_images)

# %%
print("Frame 1")
print(yolo_boxes[0])
print(yolo_categories[0])
print(yolo_scores[0])
print("Frame 2")
print(yolo_boxes[1])
print(yolo_categories[1])
print(yolo_scores[1])

# %% [markdown]
# ## 2) Hungarian Algorithm with Deep Association Metrics

# %% [markdown]
# ### 2.1 Load the Siamese Network

# %%
encoder = torch.load("model640.pt", map_location=torch.device('cpu'))
encoder = encoder.eval()

# %% [markdown]
# ### 2.2 Crop the obstacles

# %%
def crop_frames(frame, boxes):
    try:
        transforms = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(), torchvision.transforms.Resize((128,128)), torchvision.transforms.ToTensor()])
        crops = []
        crops_pytorch = []
        for box in boxes:
            crop = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
            crops.append(crop)
            crops_pytorch.append(transforms(crop))
        return crops, torch.stack(crops_pytorch)
    except:
        return [],[]

# %%
from scipy.stats import multivariate_normal

def get_gaussian_mask():
	#128 is image size
	x, y = np.mgrid[0:1.0:128j, 0:1.0:128j]
	xy = np.column_stack([x.flat, y.flat])
	mu = np.array([0.5,0.5])
	sigma = np.array([0.22,0.22])
	covariance = np.diag(sigma**2)
	z = multivariate_normal.pdf(xy, mean=mu, cov=covariance)
	z = z.reshape(x.shape)

	z = z / z.max()
	z  = z.astype(np.float32)

	mask = torch.from_numpy(z)

	return mask
gaussian_mask = get_gaussian_mask()

# %% [markdown]
# #### Frame 1

# %%
pics = copy.deepcopy(images)

# %%
print(pics[0].shape)
h, w, _ = pics[0].shape

# %%
crops_1, crops_pytorch_1 = crop_frames(pics[0], yolo_boxes[0])
crops_pytorch_1 = gaussian_mask * crops_pytorch_1
visualize_images(crops_1)

# %% [markdown]
# #### Frame 2

# %%
print(pics[1].shape)
h, w, _ = pics[1].shape

# %%
crops_2, crops_pytorch_2 = crop_frames(pics[1], yolo_boxes[1])
crops_pytorch_2 = gaussian_mask * crops_pytorch_2
visualize_images(crops_2)

# %% [markdown]
# ### 2.3 Cosine Distance

# %%
def cosine_similarity(a, b, data_is_normalized=False):
    if not data_is_normalized:
        a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
        b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
    return np.dot(a, b.T)

# %%
## EXAMPLE
a = np.array([[0,0,1]])
b = np.array([[0,1,1]])
c = np.array([[1,1,0]])

print(cosine_similarity(a,b))
print(cosine_similarity(a,c))
print(cosine_similarity(a,a))

# %%
def get_features(encoder, processed_crops):
    features = []
    if len(processed_crops)>0:
        features = encoder.forward_once(processed_crops)
        features = features.detach().cpu().numpy()
        if len(features.shape)==1:
            features = np.expand_dims(features,0)
    return features  

# %%
features_1 = get_features(encoder, crops_pytorch_1)
features_2 = get_features(encoder, crops_pytorch_2) 

# %%
visualize_images([crops_1[0], crops_2[2]])

# %%
for a, box1 in enumerate(yolo_boxes[0]):
    print("--")
    for b, box2 in enumerate(yolo_boxes[1]):
        dist = cosine_similarity(features_1[a].reshape(1,1024), features_2[b].reshape(1,1024))
        print("Box ", a , ": ", str(box1),"Box ", b, ": ", str(box2), "Distance: ", str(dist[0][0]))

# %% [markdown]
# ### 2.4 Adding the other costs

# %%
def box_iou(box1, box2, w = 1280, h=360):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    inter_area = max(0, xB - xA + 1) * max(0, yB - yA + 1) #abs((xi2 - xi1)*(yi2 - yi1))
    # Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1) #abs((box1[3] - box1[1])*(box1[2]- box1[0]))
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1) #abs((box2[3] - box2[1])*(box2[2]- box2[0]))
    union_area = (box1_area + box2_area) - inter_area
    # IoU
    iou = inter_area/float(union_area)
    return iou

# %%
def check_division_by_0(value, epsilon=0.01):
    if value < epsilon:
        value = epsilon
    return value

# %%
def sanchez_matilla(box1, box2, w = 1280, h=360):
    Q_dist = sqrt(pow(w,2)+pow(h,2))
    Q_shape = w*h
    distance_term = Q_dist/check_division_by_0(sqrt(pow(box1[0] - box2[0], 2)+pow(box1[1] -box2[1],2)))
    shape_term = Q_shape/check_division_by_0(sqrt(pow(box1[2] - box2[2], 2)+pow(box1[3] - box2[3],2)))
    linear_cost = distance_term*shape_term
    return linear_cost

# %%
from math import sqrt, exp

def yu(box1, box2):
    w1 = 0.5
    w2 = 1.5
    a= (box1[0] - box2[0])/check_division_by_0(box1[2])
    a_2 = pow(a,2)
    b = (box1[1] - box2[1])/check_division_by_0(box1[3])
    b_2 = pow(b,2)
    ab = (a_2+b_2)*w1*(-1)
    c = abs(box1[3] - box2[3])/(box1[3]+box2[3])
    d = abs(box1[2]-box2[2])/(box1[2]+box2[2])
    cd = (c+d)*w2*(-1)
    exponential_cost = exp(ab)*exp(cd)
    return exponential_cost

# %% [markdown]
# ### 2.5. Main Function

# %%
def total_cost(old_box, new_box, old_features, new_features, iou_thresh = 0.3, linear_thresh = 10000, exp_thresh = 0.5, feat_thresh = 0.2):
    iou_cost = box_iou(old_box, new_box)
    linear_cost = sanchez_matilla(old_box, new_box, w= 1920, h=1080)
    exponential_cost = yu(old_box, new_box)
    feature_cost = cosine_similarity(old_features, new_features)[0][0]

    if (iou_cost >= iou_thresh and linear_cost >= linear_thresh and exponential_cost>=exp_thresh and feature_cost >= feat_thresh):
        return iou_cost

    else:
        return 0

# %% [markdown]
# #### Association

# %%
from scipy.optimize import linear_sum_assignment

def associate(old_boxes, new_boxes, old_features, new_features):
    """
    old_boxes will represent the former bounding boxes (at time 0)
    new_boxes will represent the new bounding boxes (at time 1)
    Function goal: Define a Hungarian Matrix with IOU as a metric and return, for each box, an id
    """
    if len(old_boxes)==0 and len(new_boxes)==0:
        return [], [], []
    elif(len(old_boxes)==0):
        return [], [i for i in range(len(new_boxes))], []
    elif(len(new_boxes)==0):
        return [], [], [i for i in range(len(old_boxes))]
        
    # Define a new IOU Matrix nxm with old and new boxes
    iou_matrix = np.zeros((len(old_boxes),len(new_boxes)),dtype=np.float32)
    
    # Go through boxes and store the IOU value for each box 
    for i,old_box in enumerate(old_boxes):
        for j,new_box in enumerate(new_boxes):
            iou_matrix[i][j] = total_cost(old_box, new_box, old_features[i].reshape(1,1024), new_features[j].reshape(1,1024))

    #print(iou_matrix)
    # Call for the Hungarian Algorithm
    hungarian_row, hungarian_col = linear_sum_assignment(-iou_matrix)
    hungarian_matrix = np.array(list(zip(hungarian_row, hungarian_col)))

    # Create new unmatched lists for old and new boxes
    matches, unmatched_detections, unmatched_trackers = [], [], []

    #print(hungarian_matrix)
    
    # Go through old boxes, if no matched detection, add it to the unmatched_old_boxes
    for t,trk in enumerate(old_boxes):
	    if(t not in hungarian_matrix[:,0]):
		    unmatched_trackers.append(t)
    
    # Go through new boxes, if no matched tracking, add it to the unmatched_new_boxes
    for d, det in enumerate(new_boxes):
        if(d not in hungarian_matrix[:,1]):
                unmatched_detections.append(d)
    
    # Go through the Hungarian Matrix, if matched element has IOU < threshold (0.3), add it to the unmatched 
    for h in hungarian_matrix:
        if(iou_matrix[h[0],h[1]]<0.3):
            unmatched_trackers.append(h[0]) # Return INDICES directly
            unmatched_detections.append(h[1]) # Return INDICES directly
        else:
            matches.append(h.reshape(1,2))
    
    if(len(matches)==0):
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches,axis=0)
    

    return matches, unmatched_detections,unmatched_trackers

# %%
matches, unmatched_detections, unmatched_trackers = associate(yolo_boxes[0], yolo_boxes[1], features_1, features_2)

# %%
print(yolo_boxes[0])
print(yolo_boxes[1])
print(matches)

# %%
print(unmatched_detections)

# %%
print(unmatched_trackers)

# %% [markdown]
# ## 3) Hungarian Tracking Loop

# %%
MIN_HIT_STREAK = 1
MAX_UNMATCHED_AGE = 1

# %%
class Obstacle():
    def __init__(self, idx, box, features=None,  age=1, unmatched_age=0):
        """
        Init function. The obstacle must have an id and a box.
        """
        self.idx = idx
        self.box = box
        self.features = features
        self.age = age
        self.unmatched_age = unmatched_age

# %%
def main(input_image):
    global stored_obstacles
    global idx

    # 1 — Run Obstacle Detection & Convert the Boxes
    final_image = copy.deepcopy(input_image)
    h, w, _ = final_image.shape

    _, out_boxes, _, _ = inference(input_image)
    crops, crops_pytorch = crop_frames(final_image, out_boxes)
    features = get_features(encoder, crops_pytorch)
    
    print("----> New Detections: ", out_boxes)
    # Define the list we'll return:
    new_obstacles = []

    old_obstacles = [obs.box for obs in stored_obstacles] # Simply get the boxes
    old_features = [obs.features for obs in stored_obstacles]
    
    matches, unmatched_detections, unmatched_tracks = associate(old_obstacles, out_boxes, old_features, features)

    # Matching
    for match in matches:
        obs = Obstacle(stored_obstacles[match[0]].idx, out_boxes[match[1]], features[match[1]], stored_obstacles[match[0]].age +1)
        new_obstacles.append(obs)
        print("Obstacle ", obs.idx, " with box: ", obs.box, "has been matched with obstacle ", stored_obstacles[match[0]].box, "and now has age: ", obs.age)
    
    # New (Unmatched) Detections
    for d in unmatched_detections:
        obs = Obstacle(idx, out_boxes[d], features[d])
        new_obstacles.append(obs)
        idx+=1
        print("Obstacle ", obs.idx, " has been detected for the first time: ", obs.box)

    # Unmatched Tracks
    for t in unmatched_tracks:
        i = old_obstacles.index(stored_obstacles[t].box)
        print("Old Obstacles tracked: ", stored_obstacles[i].box)
        if i is not None:
            obs = stored_obstacles[i]
            obs.unmatched_age +=1
            new_obstacles.append(obs)
            print("Obstacle ", obs.idx, "is a long term obstacle unmatched ", obs.unmatched_age, "times.")

    # Draw the Boxes
    for i, obs in enumerate(new_obstacles):
        if obs.unmatched_age > MAX_UNMATCHED_AGE:
            new_obstacles.remove(obs)

        if obs.age >= MIN_HIT_STREAK:
            left, top, right, bottom = obs.box
            cv2.rectangle(final_image, (left, top), (right, bottom), id_to_color(obs.idx*10), thickness=7)
            final_image = cv2.putText(final_image, str(obs.idx),(left - 10,top - 10),cv2.FONT_HERSHEY_SIMPLEX, 1,id_to_color(obs.idx*10),thickness=4)

    stored_obstacles = new_obstacles

    return final_image, stored_obstacles

# %%
### Calling the main loop

fig=plt.figure(figsize=(100,100))
out_imgs = []
idx = 0
stored_obstacles = []

for i in range(len(images)):
    out_img, stored_obstacles = main(images[i])
    out_imgs.append(out_img)
    fig.add_subplot(1, len(images), i+1)
    plt.imshow(out_imgs[i])

plt.show()

# %% [markdown]
# Test Again with more images

# %%
images = []
index = 34

# 100 — Same number of detections
# 300 — 2 Lost Tracks
# 782 — 1 New Detection
# 654 — Edge Case

for img in images_files[index-2:index+10]:
    images.append(cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB))

# %%
### Call the main loop
idx = 0
stored_obstacles = []
fig=plt.figure(figsize=(100,100))
pics = copy.deepcopy(images)
out_imgs = []

for i in range(len(pics)):
    out_img, stored_obstacles = main(pics[i])
    out_imgs.append(out_img)
    fig.add_subplot(1, len(pics), i+1)
    plt.imshow(out_imgs[i])

plt.show()

# %% [markdown]
# # Video

# %%
idx = 0
stored_obstacles = []

video_images = images_files
result_video = []

for img in tqdm(video_images):
    out_img, stored_obstacles = main(cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB))
    if out_img is not None:
        result_video.append(out_img)

# %%
out = cv2.VideoWriter('output_deepsort.mp4',cv2.VideoWriter_fourcc(*'MP4V'), 15, (out_img.shape[1],out_img.shape[0]))

for img in result_video:
  out.write(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
out.release()

# %%




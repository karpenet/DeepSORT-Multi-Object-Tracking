from math import exp
import cv2
import torch
import torchvision

from utils.math_utils import check_division_by_0

def draw_bboxes(image, boxes, categories, classes, mot_mode=False):
    """
    Draw bounding boxes on the image.

    Args:
        image (numpy.ndarray): The image on which to draw.
        boxes (list): List of bounding boxes.
        categories (list): List of category indices corresponding to the boxes.
        classes (list): List of class names.
        mot_mode (bool): If True, use different colors for each box.

    Returns:
        numpy.ndarray: Image with bounding boxes drawn.
    """
    h, w, _ = image.shape
    for i, box in enumerate(boxes):
        label = classes[int(categories[i])]
        color = map_id_to_bbox_color(i * 10) if mot_mode else (255, 0, 0)
        cv2.rectangle(
            image,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            color,
            thickness=min(h, w)//200,
        )
        cv2.putText(
            image,
            str(label),
            (int(box[0]), int(box[1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            thickness=min(h, w)//100,
        )
    return image

def map_id_to_bbox_color(idx):
    """
    Convert an id to a color.

    Args:
        idx (int): The id to convert.

    Returns:
        tuple: A tuple representing the color (red, green, blue).
    """
    blue = idx * 5 % 256
    green = idx * 12 % 256
    red = idx * 23 % 256
    return (red, green, blue)

def crop_frames(frame, boxes):
    """
    Crop frames based on bounding boxes.

    Args:
        frame (numpy.ndarray): The frame to crop.
        boxes (list): List of bounding boxes.

    Returns:
        tuple: List of cropped images and a tensor of cropped images.
    """
    try:
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.Resize((128, 128)),
                torchvision.transforms.ToTensor(),
            ]
        )
        crops = []
        crops_pytorch = []
        for box in boxes:
            crop = frame[int(box[1]) : int(box[3]), int(box[0]) : int(box[2])]
            crops.append(crop)
            crops_pytorch.append(transforms(crop))
        return crops, torch.stack(crops_pytorch)
    except:
        return [], []

def yu(box1, box2):
    """
    Calculate the exponential cost between two bounding boxes.

    Args:
        box1 (list): First bounding box.
        box2 (list): Second bounding box.

    Returns:
        float: The exponential cost.
    """
    w1 = 0.5
    w2 = 1.5
    a = (box1[0] - box2[0]) / check_division_by_0(box1[2])
    a_2 = pow(a, 2)
    b = (box1[1] - box2[1]) / check_division_by_0(box1[3])
    b_2 = pow(b, 2)
    ab = (a_2 + b_2) * w1 * (-1)
    c = abs(box1[3] - box2[3]) / (box1[3] + box2[3])
    d = abs(box1[2] - box2[2]) / (box1[2] + box2[2])
    cd = (c + d) * w2 * (-1)
    exponential_cost = exp(ab) * exp(cd)
    return exponential_cost

def compute_iou(box1, box2, w=1280, h=360):
    """
    Compute the Intersection over Union (IoU) of two bounding boxes.

    Args:
        box1 (list): First bounding box.
        box2 (list): Second bounding box.
        w (int): Width of the image.
        h (int): Height of the image.

    Returns:
        float: The IoU value.
    """
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    inter_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    union_area = (box1_area + box2_area) - inter_area
    iou = inter_area / float(union_area)
    return iou
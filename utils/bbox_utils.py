import cv2
import torch
import torchvision
from typing import List, Tuple
from utils.typing import Image, BBox

def draw_bboxes(image: Image, boxes: List[BBox], categories: List[int], classes: List[str], mot_mode: bool = False) -> Image:
    h, w, _ = image.shape
    for i, box in enumerate(boxes):
        label = classes[int(categories[i])]
        color = map_id_to_bbox_color(i * 10) if mot_mode else (255, 0, 0)
        cv2.rectangle(
            image,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            color,
            thickness=min(h, w) // 200,
        )
        cv2.putText(
            image,
            str(label),
            (int(box[0]), int(box[1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            thickness=min(h, w) // 100,
        )
    return image

def map_id_to_bbox_color(idx: int) -> Tuple[int, int, int]:
    blue = idx * 5 % 256
    green = idx * 12 % 256
    red = idx * 23 % 256
    return (red, green, blue)

def crop_frames(frame: Image, boxes: List[BBox]) -> Tuple[List[Image], torch.Tensor]:
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

def compute_iou(box1: BBox, box2: BBox, w: int = 1280, h: int = 360) -> float:
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    inter_area = max(0, xB - xA + 1) * max(0, yA - yA + 1)
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    union_area = (box1_area + box2_area) - inter_area
    iou = inter_area / float(union_area)
    return iou

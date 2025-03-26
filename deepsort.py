import copy
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment
from utils.bbox_utils import crop_frames, draw_bboxes, compute_iou, map_id_to_bbox_color
from utils.math_utils import sanchez_matilla, cosine_similarity, yu
from utils.typing import Image, BBoxList, BBox, List, Tuple, Tensor, FeatureVector
from utils.logging import Logger
from transformers import CLIPProcessor, CLIPModel

# Constants for tracking
MIN_HIT_STREAK = 1
MAX_UNMATCHED_AGE = 1


class Detection:
    def __init__(
        self,
        idx: int,
        category: int,
        box: BBox,
        features: np.ndarray = None,
        age: int = 1,
        unmatched_age: int = 0,
    ) -> None:
        """
        Initialize a detection object.

        Args:
            idx (int): Unique identifier for the detection.
            box (list): Bounding box coordinates [x1, y1, x2, y2].
            features (np.array): Feature vector for the detection.
            age (int): Age of the detection.
            unmatched_age (int): Number of frames the detection has been unmatched.
        """
        self.idx = idx
        self.category = category
        self.box = box
        self.features = features
        self.age = age
        self.unmatched_age = unmatched_age


class DeepSort:
    def __init__(self) -> None:
        """
        Initialize the DeepSort object with a YOLOv5 detector and a Siamese network encoder.
        """
        self.logger = Logger()
        # Load object detector
        self.detector = YOLO("models/yolov8n.engine", task="detect")
        self.tracking_classes = [1, 2, 3, 5, 7]
        # self.tracking_classes = [0]
        with open("coco.names", "rt") as f:
            self.detector_classes = f.read().rstrip("\n").split("\n")

        # # Load CLIP Encoder
        # self.encoder = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        # self.preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        # Load Siamese Network
        self.encoder = torch.load(
            "models/model640.pt", map_location=torch.device("cpu")
        )
        self.encoder = torch.compile(self.encoder).to('cuda')
        self.encoder.eval()
        self.stored_obstacles = []
        self.idx = 0

    def compute_cost(
        self,
        old_box: BBox,
        new_box: BBox,
        old_features: np.ndarray,
        new_features: np.ndarray,
        img_size: tuple = (1920, 1080),
        iou_thresh: float = 0.3,
        linear_thresh: float = 10000,
        exp_thresh: float = 0.5,
        feat_thresh: float = 0.2,
    ) -> float:
        """
        Compute the cost between old and new detections using various metrics.

        Args:
            old_box (list): Bounding box coordinates of the old detection.
            new_box (list): Bounding box coordinates of the new detection.
            old_features (np.array): Feature vector of the old detection.
            new_features (np.array): Feature vector of the new detection.
            iou_thresh (float): IOU threshold.
            linear_thresh (float): Linear distance threshold.
            exp_thresh (float): Exponential distance threshold.
            feat_thresh (float): Feature similarity threshold.

        Returns:
            float: Cost value based on the metrics.
        """
        iou_cost = compute_iou(old_box, new_box)
        linear_cost = sanchez_matilla(old_box, new_box, w=img_size[0], h=img_size[1])
        exponential_cost = yu(old_box, new_box)
        feature_cost = cosine_similarity(old_features, new_features)[0][0]

        self.logger.log(
            f"Computed costs - IOU: {iou_cost}, Linear: {linear_cost}, Exponential: {exponential_cost}, Feature: {feature_cost}"
        )

        if (
            iou_cost >= iou_thresh
            and linear_cost >= linear_thresh
            and exponential_cost >= exp_thresh
            and feature_cost >= feat_thresh
        ):
            return iou_cost
        return 0

    def get_clip_features(self, processed_crops: Tensor) -> FeatureVector:
        # Load and preprocess the image
        # Preprocess all images in the batch
        images = [crop for crop in processed_crops]
        image_input = self.preprocess(images=images, return_tensors="pt", padding=True)

        # Compute image embeddings
        features = []
        if len(processed_crops) > 0:
            with torch.no_grad():
                features = self.encoder.get_image_features(**image_input)
                features = features.detach().cpu().numpy()
                if len(features.shape) == 1:
                    features = np.expand_dims(features, 0)
            self.logger.log(f"Computed encoder features: {features}")
        return features

    def get_encoder_features(self, processed_crops: Tensor) -> FeatureVector:
        """
        Get features from the encoder for the given crops.

        Args:
            processed_crops (Tensor): Cropped images to be processed by the encoder.

        Returns:
            Vector: Feature vectors for the crops.
        """
        features = []
        if len(processed_crops) > 0:
            features = self.encoder.forward_once(processed_crops)
            features = features.detach().cpu().numpy()
            if len(features.shape) == 1:
                features = np.expand_dims(features, 0)
        self.logger.log(f"Computed encoder features: {features}")
        return features

    def detector_inference(
        self, frame: Image
    ) -> Tuple[np.ndarray, BBoxList, List[int], List[float]]:
        """
        Perform object detection on the given frame.

        Args:
            frame (np.array): Input image frame.

        Returns:
            tuple: Processed image with bounding boxes, list of bounding boxes, list of categories, list of scores.
        """
        print(frame.shape)
        results = self.detector.predict(frame, device="cuda", conf=0.5, iou=0.4, classes=self.tracking_classes, half=True)
        predictions = results[0]
        boxes = predictions.boxes.xyxy
        boxes_int = [[int(v) for v in box] for box in boxes]
        scores = predictions.boxes.conf
        categories = predictions.boxes.cls
        categories_int = [int(c) for c in categories]
        img_out = draw_bboxes(frame, boxes_int, categories_int, self.detector_classes)

        self.logger.log(
            f"Detector inference results - Boxes: {boxes_int}, Categories: {categories_int}, Scores: {scores}"
        )

        return img_out, boxes_int, categories_int, scores

    def _get_boxes(self, detections: List[Detection]):
        return [detection.box for detection in detections]

    def _get_features(self, detections: List[Detection]):
        return [detection.features for detection in detections]

    def _get_idx(self, detections: List[Detection]):
        return [detection.idx for detection in detections]

    def associate(
        self, new_detections: List[Detection], img_size: tuple = (1920, 1080),
    ) -> Tuple[np.ndarray, List[int], List[int]]:
        """
        Associate old and new detections using the Hungarian algorithm.

        Args:
            old_boxes (list): List of old bounding boxes.
            new_boxes (list): List of new bounding boxes.
            old_features (list): List of old feature vectors.
            new_features (list): List of new feature vectors.

        Returns:
            tuple: Matched indices, unmatched new detections, unmatched old detections.
        """
        old_boxes = self._get_boxes(self.stored_obstacles)
        new_boxes = self._get_boxes(new_detections)

        old_features = self._get_features(self.stored_obstacles)
        new_features = self._get_features(new_detections)

        if len(old_boxes) == 0 and len(new_boxes) == 0:
            return [], [], []
        elif len(old_boxes) == 0:
            return [], [i for i in range(len(new_boxes))], []
        elif len(new_boxes) == 0:
            return [], [], [i for i in range(len(old_boxes))]

        # Define a new IOU Matrix nxm with old and new boxes
        iou_matrix = np.zeros((len(old_boxes), len(new_boxes)), dtype=np.float32)

        # Go through boxes and store the IOU value for each box
        for i, old_box in enumerate(old_boxes):
            for j, new_box in enumerate(new_boxes):
                iou_matrix[i][j] = self.compute_cost(
                    old_box,
                    new_box,
                    old_features[i].reshape(1, 1024),
                    new_features[j].reshape(1, 1024),
                    img_size=img_size,
                )

        self.logger.log(f"IOU Matrix: {iou_matrix}")

        # Call for the Hungarian Algorithm
        hungarian_row, hungarian_col = linear_sum_assignment(-iou_matrix)
        hungarian_matrix = np.array(list(zip(hungarian_row, hungarian_col)))

        self.logger.log(
            f"Hungarian algorithm results - Rows: {hungarian_row}, Columns: {hungarian_col}"
        )

        # Create new unmatched lists for old and new boxes
        matches, unmatched_detections, unmatched_trackers = [], [], []

        # Go through old boxes, if no matched detection, add it to the unmatched_old_boxes
        for t, trk in enumerate(old_boxes):
            if t not in hungarian_matrix[:, 0]:
                unmatched_trackers.append(t)

        # Go through new boxes, if no matched tracking, add it to the unmatched_new_boxes
        for d, det in enumerate(new_boxes):
            if d not in hungarian_matrix[:, 1]:
                unmatched_detections.append(d)

        # Go through the Hungarian Matrix, if matched element has IOU < threshold (0.3), add it to the unmatched
        for h in hungarian_matrix:
            if iou_matrix[h[0], h[1]] < 0.3:
                unmatched_trackers.append(h[0])  # Return INDICES directly
                unmatched_detections.append(h[1])  # Return INDICES directly
            else:
                matches.append(h.reshape(1, 2))

        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)

        self.logger.log(
            f"Association results - Matches: {matches}, Unmatched Detections: {unmatched_detections}, Unmatched Trackers: {unmatched_trackers}"
        )

        return matches, unmatched_detections, unmatched_trackers

    def register_detections(
        self, categories: List[int], bboxes: List[BBox], features: List[np.ndarray]
    ) -> List[Detection]:
        return [
            Detection(idx=-1, category=categories[i], box=bboxes[i], features=features[i])
            for i in range(len(bboxes))
        ]

    def inference(self, frame: Image) -> Tuple[np.ndarray, List[Detection]]:
        """
        Perform inference on the given frame to detect and track objects.

        Args:
            frame (np.array): Input image frame.

        Returns:
            tuple: Processed image with bounding boxes, list of stored obstacles.
        """
        # 1 — Run Obstacle Detection & Convert the Boxes
        final_image = copy.deepcopy(frame)
        h, w, _ = final_image.shape

        _, out_boxes, out_categories, _ = self.detector_inference(frame)
        crops, crops_pytorch = crop_frames(final_image, out_boxes)
        features = self.get_encoder_features(crops_pytorch)

        self.logger.log(f"New Detections: {out_boxes}")

        # Define the list we'll return:
        new_obstacles = []
        new_detections = self.register_detections(out_categories, out_boxes, features)

        old_obstacles = [
            obs.box for obs in self.stored_obstacles
        ]  # Simply get the boxes
        old_features = [obs.features for obs in self.stored_obstacles]

        matches, unmatched_detections, unmatched_tracks = self.associate(new_detections, img_size=(w, h))

        # Matching
        for match in matches:
            obs = Detection(
                self.stored_obstacles[match[0]].idx,
                self.stored_obstacles[match[0]].category,
                out_boxes[match[1]],
                features[match[1]],
                self.stored_obstacles[match[0]].age + 1,
            )
            new_obstacles.append(obs)
            self.logger.log(
                f"Obstacle {obs.idx} with box: {obs.box} has been matched with obstacle {self.stored_obstacles[match[0]].box} and now has age: {obs.age}"
            )

        # New (Unmatched) Detections
        for d in unmatched_detections:
            obs = Detection(
                self.idx, new_detections[d].category, out_boxes[d], features[d]
            )
            new_obstacles.append(obs)
            self.idx += 1
            self.logger.log(
                f"Obstacle {obs.idx} has been detected for the first time: {obs.box}"
            )

        # Unmatched Tracks
        for t in unmatched_tracks:
            i = old_obstacles.index(self.stored_obstacles[t].box)
            self.logger.log(f"Old Obstacles tracked: {self.stored_obstacles[i].box}")
            if i is not None:
                obs = self.stored_obstacles[i]
                obs.unmatched_age += 1
                new_obstacles.append(obs)
                self.logger.log(
                    f"Obstacle {obs.idx} is a long term obstacle unmatched {obs.unmatched_age} times."
                )
        
        # Draw the Boxes
        for i, obs in enumerate(new_obstacles):
            if obs.unmatched_age > MAX_UNMATCHED_AGE:
                new_obstacles.remove(obs)

            if obs.age >= MIN_HIT_STREAK:
                left, top, right, bottom = obs.box
                cv2.rectangle(
                    final_image,
                    (left, top),
                    (right, bottom),
                    map_id_to_bbox_color(obs.idx * 10),
                    thickness=7,
                )
                final_image = cv2.putText(
                    final_image,
                    f"{self.detector_classes[obs.category]}: {str(obs.idx)}",
                    (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    map_id_to_bbox_color(obs.idx * 10),
                    thickness=4,
                )        

        self.stored_obstacles = new_obstacles

        self.logger.log(
            "Final image with bounding boxes drawn and stored obstacles updated."
        )

        return final_image, self.stored_obstacles

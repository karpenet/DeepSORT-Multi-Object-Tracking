import copy
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment
from utils.bbox_utils import crop_frames, draw_bboxes, compute_iou, map_id_to_bbox_color
from utils.math_utils import sanchez_matilla, cosine_similarity, yu

# Constants for tracking
MIN_HIT_STREAK = 1
MAX_UNMATCHED_AGE = 1


class Detection:
    def __init__(self, idx, box, features=None, age=1, unmatched_age=0):
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
        self.box = box
        self.features = features
        self.age = age
        self.unmatched_age = unmatched_age


class DeepSort:
    def __init__(self) -> None:
        """
        Initialize the DeepSort object with a YOLOv5 detector and a Siamese network encoder.
        """
        # Load object detector
        self.detector = YOLO("models/yolov8n.engine", task="detect")
        with open("coco.names", "rt") as f:
            self.detector_classes = f.read().rstrip("\n").split("\n")

        # Load Siamese Network
        self.encoder = torch.load(
            "models/model640.pt", map_location=torch.device("cpu")
        )
        self.encoder.eval()
        self.stored_obstacles = []
        self.idx = 0

    def compute_cost(
        self,
        old_box,
        new_box,
        old_features,
        new_features,
        iou_thresh=0.3,
        linear_thresh=10000,
        exp_thresh=0.5,
        feat_thresh=0.2,
    ):
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
        linear_cost = sanchez_matilla(old_box, new_box, w=1920, h=1080)
        exponential_cost = yu(old_box, new_box)
        feature_cost = cosine_similarity(old_features, new_features)[0][0]

        if (
            iou_cost >= iou_thresh
            and linear_cost >= linear_thresh
            and exponential_cost >= exp_thresh
            and feature_cost >= feat_thresh
        ):
            return iou_cost
        else:
            return 0

    def get_encoder_features(self, processed_crops):
        """
        Get features from the encoder for the given crops.

        Args:
            processed_crops (torch.Tensor): Cropped images to be processed by the encoder.

        Returns:
            np.array: Feature vectors for the crops.
        """
        features = []
        if len(processed_crops) > 0:
            features = self.encoder.forward_once(processed_crops)
            features = features.detach().cpu().numpy()
            if len(features.shape) == 1:
                features = np.expand_dims(features, 0)
        return features

    def detector_inference(self, frame):
        """
        Perform object detection on the given frame.

        Args:
            frame (np.array): Input image frame.

        Returns:
            tuple: Processed image with bounding boxes, list of bounding boxes, list of categories, list of scores.
        """
        results = self.detector.predict(frame, device="cuda", conf=0.5, iou=0.4)
        predictions = results[0]
        boxes = predictions.boxes.xywh
        boxes_int = [[int(v) for v in box] for box in boxes]
        scores = predictions.boxes.conf
        categories = predictions.boxes.cls
        categories_int = [int(c) for c in categories]
        img_out = draw_bboxes(
            frame, boxes_int, categories_int, self.detector_classes, mot_mode=True
        )

        return img_out, boxes_int, categories_int, scores

    def associate(self, old_boxes, new_boxes, old_features, new_features):
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
                )

        # Call for the Hungarian Algorithm
        hungarian_row, hungarian_col = linear_sum_assignment(-iou_matrix)
        hungarian_matrix = np.array(list(zip(hungarian_row, hungarian_col)))

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

        return matches, unmatched_detections, unmatched_trackers

    def inference(self, frame):
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

        _, out_boxes, _, _ = self.detector_inference(frame)
        crops, crops_pytorch = crop_frames(final_image, out_boxes)
        features = self.get_encoder_features(crops_pytorch)

        print("----> New Detections: ", out_boxes)
        # Define the list we'll return:
        new_obstacles = []

        old_obstacles = [
            obs.box for obs in self.stored_obstacles
        ]  # Simply get the boxes
        old_features = [obs.features for obs in self.stored_obstacles]

        matches, unmatched_detections, unmatched_tracks = self.associate(
            old_obstacles, out_boxes, old_features, features
        )

        # Matching
        for match in matches:
            obs = Detection(
                self.stored_obstacles[match[0]].idx,
                out_boxes[match[1]],
                features[match[1]],
                self.stored_obstacles[match[0]].age + 1,
            )
            new_obstacles.append(obs)
            print(
                "Obstacle ",
                obs.idx,
                " with box: ",
                obs.box,
                "has been matched with obstacle ",
                self.stored_obstacles[match[0]].box,
                "and now has age: ",
                obs.age,
            )

        # New (Unmatched) Detections
        for d in unmatched_detections:
            obs = Detection(self.idx, out_boxes[d], features[d])
            new_obstacles.append(obs)
            self.idx += 1
            print(
                "Obstacle ", obs.idx, " has been detected for the first time: ", obs.box
            )

        # Unmatched Tracks
        for t in unmatched_tracks:
            i = old_obstacles.index(self.stored_obstacles[t].box)
            print("Old Obstacles tracked: ", self.stored_obstacles[i].box)
            if i is not None:
                obs = self.stored_obstacles[i]
                obs.unmatched_age += 1
                new_obstacles.append(obs)
                print(
                    "Obstacle ",
                    obs.idx,
                    "is a long term obstacle unmatched ",
                    obs.unmatched_age,
                    "times.",
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
                    str(obs.idx),
                    (left - 10, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    map_id_to_bbox_color(obs.idx * 10),
                    thickness=4,
                )

        self.stored_obstacles = new_obstacles

        return final_image, self.stored_obstacles

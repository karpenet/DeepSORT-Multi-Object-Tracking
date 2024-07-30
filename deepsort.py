import copy
import cv2
import numpy as np
import torch
import yolov5
from scipy.optimize import linear_sum_assignment
from utils.bbox_utils import crop_frames, draw_boxes_v5, box_iou, id_to_color, yu
from utils.math_utils import sanchez_matilla, cosine_similarity


MIN_HIT_STREAK = 1
MAX_UNMATCHED_AGE = 1


class Detection:
    def __init__(self, idx, box, features=None, age=1, unmatched_age=0):
        """
        Init function. The detection must have an id and a box.
        """
        self.idx = idx
        self.box = box
        self.features = features
        self.age = age
        self.unmatched_age = unmatched_age

class DeepSort:
    def __init__(self) -> None:
        # Load object detector
        self.detector = yolov5.load("models/yolov5s.pt")
        self.detector.conf = 0.5
        self.detector.iou = 0.4
        classesFile = "coco.names"
        with open(classesFile, "rt") as f:
            self.detector_classes = f.read().rstrip("\n").split("\n")

        # Load Siamese Network
        self.encoder = torch.load(
            "models/model640.pt", map_location=torch.device("cpu")
        )
        self.encoder = self.encoder.eval()
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
        iou_cost = box_iou(old_box, new_box)
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
        features = []
        if len(processed_crops) > 0:
            features = self.encoder.forward_once(processed_crops)
            features = features.detach().cpu().numpy()
            if len(features.shape) == 1:
                features = np.expand_dims(features, 0)
        return features

    def detector_inference(self, frame):
        results = self.detector(frame)
        predictions = results.pred[0]
        boxes = predictions[:, :4].tolist()
        boxes_int = [[int(v) for v in box] for box in boxes]
        scores = predictions[:, 4].tolist()
        categories = predictions[:, 5].tolist()
        categories_int = [int(c) for c in categories]
        img_out = draw_boxes_v5(frame, boxes_int, categories_int, self.detector_classes, mot_mode=True)
        return img_out, boxes_int, categories_int, scores

    def associate(self, old_boxes, new_boxes, old_features, new_features):
        """
        old_boxes will represent the former bounding boxes (at time 0)
        new_boxes will represent the new bounding boxes (at time 1)
        Function goal: Define a Hungarian Matrix with IOU as a metric and return, for each box, an id
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

        # print(iou_matrix)
        # Call for the Hungarian Algorithm
        hungarian_row, hungarian_col = linear_sum_assignment(-iou_matrix)
        hungarian_matrix = np.array(list(zip(hungarian_row, hungarian_col)))

        # Create new unmatched lists for old and new boxes
        matches, unmatched_detections, unmatched_trackers = [], [], []

        # print(hungarian_matrix)

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
        # 1 — Run Obstacle Detection & Convert the Boxes
        final_image = copy.deepcopy(frame)
        h, w, _ = final_image.shape

        _, out_boxes, _, _ = self.detector_inference(frame)
        crops, crops_pytorch = crop_frames(final_image, out_boxes)
        features = self.get_encoder_features(crops_pytorch)

        print("----> New Detections: ", out_boxes)
        # Define the list we'll return:
        new_obstacles = []

        old_obstacles = [obs.box for obs in self.stored_obstacles]  # Simply get the boxes
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
                    id_to_color(obs.idx * 10),
                    thickness=7,
                )
                final_image = cv2.putText(
                    final_image,
                    str(obs.idx),
                    (left - 10, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    id_to_color(obs.idx * 10),
                    thickness=4,
                )

        stored_obstacles = new_obstacles

        return final_image, stored_obstacles

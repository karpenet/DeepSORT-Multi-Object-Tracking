import os
import csv

import numpy as np
from scipy.optimize import linear_sum_assignment
import pandas as pd


class_name_to_class_id = {
    "Car": 2,
    "van": 2,
    "truck": 7,
    "pedestrian": 0,
    "person": 0,
    "cyclist": 1,
    "tram": 6,
    "misc": 8,
    "DontCare": 9,
    "car_2": 2,
}


convert_filter = {2: class_name_to_class_id}
ignore_filter = {2: ["DontCare"]}


def compute_iou(box1: list, box2: list) -> float:
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    inter_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    union_area = (box1_area + box2_area) - inter_area
    iou = inter_area / float(union_area) if union_area > 0 else 0
    return iou


def save_track_data(tracks):
    data_frame = {
        "frame": [],
        "track_id": [],
        "type": [],
        "bbox_1": [],
        "bbox_2": [],
        "bbox_3": [],
        "bbox_4": [],
    }
    for frame, track in enumerate(tracks):
        for detection in track:
            data_frame["frame"].append(frame)
            data_frame["track_id"].append(detection.idx)
            data_frame["type"].append(detection.category)
            data_frame["bbox_1"].append(detection.box[0])
            data_frame["bbox_2"].append(detection.box[1])
            data_frame["bbox_3"].append(detection.box[2])
            data_frame["bbox_4"].append(detection.box[3])

    df = pd.DataFrame(data_frame)
    df.to_csv("dataset/KITTI/eval/0005.txt", sep=" ", index=False, header=False)


class EvaluateMOTS:
    def __init__(self, gt_file_path: str, tracks_file_path: str):
        self.gt_data = self.load_gt_file(gt_file_path)
        self.track_data = self.load_tracks_file(tracks_file_path)
        self.track_row = 0

    def load_gt_file(self, gt_file_path: str):
        # Read the dataset with appropriate column formatting
        column_names = [
            "frame",
            "track_id",
            "type",
            "truncated",
            "occluded",
            "alpha",
            "bbox_1",
            "bbox_2",
            "bbox_3",
            "bbox_4",
            "dim_1",
            "dim_2",
            "dim_3",
            "loc_1",
            "loc_2",
            "loc_3",
            "rotation_y",
        ]

        if not os.path.isfile(gt_file_path):
            raise FileNotFoundError(f"{gt_file_path} is not a valid file!")

        # Load the data
        data = pd.read_csv(gt_file_path, sep=" ", header=None)

        # Filter unwanted data
        data = self._preprocess_gt(data, convert_filter, ignore_filter)

        # Combine vector columns into single columns
        data.columns = column_names
        data["bbox"] = data[["bbox_1", "bbox_2", "bbox_3", "bbox_4"]].values.tolist()
        data["dimensions"] = data[["dim_1", "dim_2", "dim_3"]].values.tolist()
        data["location"] = data[["loc_1", "loc_2", "loc_3"]].values.tolist()

        # Drop the original separate columns
        data_combined = data.drop(
            columns=[
                "bbox_1",
                "bbox_2",
                "bbox_3",
                "bbox_4",
                "dim_1",
                "dim_2",
                "dim_3",
                "loc_1",
                "loc_2",
                "loc_3",
            ]
        )

        return data_combined

    def _preprocess_gt(self, data, convert_filter, ignore_filter):
        for col, ignore_list in ignore_filter.items():
            class_id = data.columns[col]
            data = data[~data[class_id].isin(ignore_list)]

        for col, mapping in convert_filter.items():
            class_id = data.columns[col]
            data[class_id] = data[class_id].map(mapping)

        return data

    def load_tracks_file(self, track_file_path: str):
        # Read the dataset with appropriate column formatting
        column_names = [
            "frame",
            "track_id",
            "type",
            "bbox_1",
            "bbox_2",
            "bbox_3",
            "bbox_4",
        ]

        if not os.path.isfile(track_file_path):
            raise FileNotFoundError(f"{track_file_path} is not a valid file!")
        # Load the data
        data = pd.read_csv(track_file_path, sep=" ", header=None)
        data.columns = column_names

        # Combine vector columns into single columns
        data["bbox"] = data[["bbox_1", "bbox_2", "bbox_3", "bbox_4"]].values.tolist()

        # Drop the original separate columns
        data_combined = data.drop(columns=["bbox_1", "bbox_2", "bbox_3", "bbox_4"])

        return data_combined

    def update_metrics_csv(self, results_file_path: str):
        file_exists = os.path.isfile(results_file_path)

        with open(results_file_path, mode="a", newline="") as file:
            metrics = self.evaluate_deepsort()
            writer = csv.DictWriter(file, fieldnames=metrics.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(metrics)

    def evaluate_deepsort(self):
        metrics = {
            "Total GT": len(self.gt_data),
            "Total Detections": len(self.track_data),
            "ID Switches": self.compute_id_switches(),
            "Mostly Tracked (MT)": self.compute_mt_ml()[0],
            "Mostly Lost (ML)": self.compute_mt_ml()[1],
            "Fragmentation": self.compute_fragmentation(),
            "False Positives (FP)": self.compute_fp_fn()[0],
            "False Negatives (FN)": self.compute_fp_fn()[1],
            "Precision": self.compute_precision_recall()[0],
            "Recall": self.compute_precision_recall()[1],
            "MOTA": self.compute_mota(),
            "MOTP": self.compute_motp(),
            "IDF1 Score": self.compute_idf1(),
        }
        return metrics

    # 1. ID Switches
    def compute_id_switches(self):
        id_switches = 0
        last_ids = {}

        for frame in sorted(self.gt_data["frame"].unique()):
            gt_ids = self.gt_data[self.gt_data["frame"] == frame]["track_id"].values
            track_ids = self.track_data[self.track_data["frame"] == frame][
                "track_id"
            ].values

            # Check if any IDs have switched for each object
            for gt_id, track_id in zip(gt_ids, track_ids):
                if gt_id in last_ids and last_ids[gt_id] != track_id:
                    id_switches += 1
                last_ids[gt_id] = track_id

        return id_switches

    # 2. Mostly Tracked & Mostly Lost
    def compute_mt_ml(
        self,
        min_track_percentage=0.8,
        max_track_percentage=0.2,
    ):
        unique_ids = self.gt_data["track_id"].unique()
        mt_count = 0
        ml_count = 0

        for obj_id in unique_ids:
            gt_frames = set(self.gt_data[self.gt_data["track_id"] == obj_id]["frame"])
            track_frames = set(
                self.track_data[self.track_data["track_id"] == obj_id]["frame"]
            )
            tracked_percentage = len(track_frames.intersection(gt_frames)) / len(
                gt_frames
            )

            if tracked_percentage >= min_track_percentage:
                mt_count += 1
            elif tracked_percentage <= max_track_percentage:
                ml_count += 1

        return mt_count, ml_count

    # 3. Fragmentation
    def compute_fragmentation(self):
        fragments = 0
        for obj_id in self.gt_data["track_id"].unique():
            gt_frames = sorted(
                self.gt_data[self.gt_data["track_id"] == obj_id]["frame"].values
            )
            track_frames = sorted(
                self.track_data[self.track_data["track_id"] == obj_id]["frame"].values
            )
            fragments += len(set(gt_frames).difference(track_frames))

        return fragments

    # 4. False Positives & False Negatives
    def compute_fp_fn(self):
        frames = sorted(self.gt_data["frame"].unique())
        fp, fn = 0, 0

        for frame in frames:
            gt_ids = self.gt_data[self.gt_data["frame"] == frame]["track_id"].values
            track_ids = self.track_data[self.track_data["frame"] == frame][
                "track_id"
            ].values

            fn += len(
                set(gt_ids) - set(track_ids)
            )  # False negatives: missed ground truth IDs
            fp += len(
                set(track_ids) - set(gt_ids)
            )  # False positives: extra IDs detected by tracker

        return fp, fn

    # 5. Tracking Precision & Recall
    def compute_precision_recall(self):
        tp = 0
        for frame in sorted(self.gt_data["frame"].unique()):
            gt_ids = self.gt_data[self.gt_data["frame"] == frame]["track_id"].values
            track_ids = self.track_data[self.track_data["frame"] == frame][
                "track_id"
            ].values

            tp += len(set(gt_ids) & set(track_ids))  # True positives
        fp, fn = self.compute_fp_fn()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        return precision, recall

    # 6. MOTA
    def compute_mota(self):
        fp, fn = self.compute_fp_fn()
        id_switches = self.compute_id_switches()
        total_gt = len(self.gt_data)

        mota = 1 - (fp + fn + id_switches) / total_gt
        return mota

    # 7. MOTP
    def compute_motp(self):
        total_iou = 0
        matches = 0

        for frame in sorted(self.gt_data["frame"].unique()):
            gt_boxes = self.gt_data[self.gt_data["frame"] == frame][["bbox"]].values.tolist()
            track_boxes = self.track_data[self.track_data["frame"] == frame][["bbox"]].values.tolist()

            gt_boxes = [gt_box[0] for gt_box in gt_boxes]
            track_boxes = [track_box[0] for track_box in track_boxes]

            if len(gt_boxes) == 0 or len(track_boxes) == 0:
                continue

            cost_matrix = np.zeros((len(gt_boxes), len(track_boxes)))
            for i, gt_box in enumerate(gt_boxes):
                for j, track_box in enumerate(track_boxes):
                    cost_matrix[i, j] = 1 - compute_iou(gt_box, track_box)

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for i, j in zip(row_ind, col_ind):
                total_iou += compute_iou(gt_boxes[i], track_boxes[j])
                matches += 1

        motp = total_iou / matches if matches > 0 else 0
        return motp

    # 8. IDF1 Score
    def compute_idf1(self):
        precision, recall = self.compute_precision_recall()
        idf1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )
        return idf1

import os
import csv
import traceback

import numpy as np
from scipy.optimize import linear_sum_assignment
from utils.bbox_utils import compute_iou
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


class EvaluateMOTS:
    def __init__(self):
        self.gt_data = self.load_gt_file()
        print(self.gt_data)
        # self.track_data = self.load_track_data_file()
        self.track_row = 0

    def load_gt_file(self):
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

        # Load the data
        data = pd.read_csv("dataset/KITTI/labels/0005.txt", sep=" ", header=None)

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

    def append_track_data(self, tracks):
        data_frame = {
            "frame": [],
            "track_id": [],
            # "type": [],
            "bbox_1": [],
            "bbox_2": [],
            "bbox_3": [],
            "bbox_4": [],
        }
        for frame, track in enumerate(tracks):
            for detection in track:
                data_frame["frame"].append(frame)
                data_frame["track_id"].append(detection.idx)
                # data_frame["type"].append(detection.det_class)
                data_frame["bbox_1"].append(detection.box[0])
                data_frame["bbox_2"].append(detection.box[1])
                data_frame["bbox_3"].append(detection.box[2])
                data_frame["bbox_4"].append(detection.box[3])
        
        df = pd.DataFrame(data_frame)
        df.to_csv("dataset/KITTI/eval/0005.txt", sep=" ", index=False, header=False)

    

    def load_tracks_file(self):
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

        # Load the data
        data = pd.read_csv("dataset/KITTI/eval/0005.txt", sep=" ", header=None)
        data.columns = column_names

        # Combine vector columns into single columns
        data["bbox"] = data[["bbox_1", "bbox_2", "bbox_3", "bbox_4"]].values.tolist()

        # Drop the original separate columns
        data_combined = data.drop(columns=["bbox_1", "bbox_2", "bbox_3", "bbox_4"])

        return data_combined

    def load_track_data_file(self):
        track_data_path = "dataset/KITTI/eval/0005.txt"
        if not os.path.exists(track_data_path):
            open(track_data_path, "w").close()
        self.track_data = pd.read_csv(
            "dataset/KITTI/eval/0005.txt", sep=" ", header=None
        )

    # def _preprocess_gt(self, df, col_values_dict):
    #     for col, ignore_list in ignore_filter.items():
    #         class_id = df.columns[col]
    #         self.gt_data = self.gt_data[~df[class_id].isin(ignore_list)]

    #     for col, mapping in convert_filter.items():
    #         class_id = self.gt_data.columns[col]
    #         self.gt_data[class_id] = self.gt_data[class_id].map(mapping)

    def evaluate_deepsort(self):
        metrics = {
            "ID Switches": compute_id_switches(self.gt_data, self.track_data),
            "Mostly Tracked (MT)": compute_mt_ml(self.gt_data, self.track_data)[0],
            "Mostly Lost (ML)": compute_mt_ml(self.gt_data, self.track_data)[1],
            "Fragmentation": compute_fragmentation(self.gt_data, self.track_data),
            "False Positives (FP)": compute_fp_fn(self.gt_data, self.track_data)[0],
            "False Negatives (FN)": compute_fp_fn(self.gt_data, self.track_data)[1],
            "Precision": compute_precision_recall(self.gt_data, self.track_data)[0],
            "Recall": compute_precision_recall(self.gt_data, self.track_data)[1],
            "MOTA": compute_mota(self.gt_data, self.track_data),
            "MOTP": compute_motp(self.gt_data, self.track_data),
            "IDF1 Score": compute_idf1(self.gt_data, self.track_data),
        }
        return metrics

    def update_metrics_csv(self, results_csv):
        metrics = self.evaluate_deepsort()
        if results_csv:
            file_exists = os.path.isfile(results_csv)
            with open(results_csv, mode="a", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=metrics.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(metrics)

    def _load_gt_text_file(
        self, valid_filter=None, crowd_ignore_filter=None, convert_filter=None
    ):
        """Function that loads data from a text file and separates detections by timestep."""

        crowd_ignore_filter = crowd_ignore_filter or {}
        convert_filter = convert_filter or {}

        read_data = {}
        crowd_ignore_data = {}

        try:
            with open(self.file) as fp:
                if fp.read(1):
                    fp.seek(0)
                    dialect = csv.Sniffer().sniff(fp.readline())
                    dialect.skipinitialspace = True
                    fp.seek(0)
                    reader = csv.reader(fp, dialect)

                    for row in reader:
                        try:
                            if row[-1] == "":
                                row = row[:-1]
                            timestep = str(int(float(row[0])))

                            is_ignored = any(
                                row[ignore_key].lower() in ignore_value
                                for ignore_key, ignore_value in crowd_ignore_filter.items()
                            )
                            if is_ignored:
                                for (
                                    convert_key,
                                    convert_value,
                                ) in convert_filter.items():
                                    row[convert_key] = convert_value[
                                        row[convert_key].lower()
                                    ]
                                crowd_ignore_data.setdefault(timestep, []).append(row)
                                continue

                            if valid_filter and any(
                                row[key].lower() not in value
                                for key, value in valid_filter.items()
                            ):
                                continue

                            if int(float(row[1])) < 0:
                                continue

                            for convert_key, convert_value in convert_filter.items():
                                row[convert_key] = convert_value[
                                    row[convert_key].lower()
                                ]

                            read_data.setdefault(timestep, []).append(row)

                        except Exception:
                            exc_str = f"In self.file {os.path.basename(self.file)} the following line cannot be read correctly: \n{' '.join(row)}"
                            raise Exception(exc_str)

        except Exception:
            print(f"Error loading self.file: {self.file}, printing traceback.")
            traceback.print_exc()
            raise Exception(
                f"self.file {os.path.basename(self.file)} cannot be read because it is either not present or invalidly formatted"
            )

        return read_data, crowd_ignore_data


# 1. ID Switches
def compute_id_switches(gt_data, track_data):
    id_switches = 0
    last_ids = {}

    for frame in sorted(gt_data["frame"].unique()):
        gt_ids = gt_data[gt_data["frame"] == frame]["id"].values
        track_ids = track_data[track_data["frame"] == frame]["id"].values

        # Check if any IDs have switched for each object
        for gt_id, track_id in zip(gt_ids, track_ids):
            if gt_id in last_ids and last_ids[gt_id] != track_id:
                id_switches += 1
            last_ids[gt_id] = track_id

    return id_switches


# 2. Mostly Tracked & Mostly Lost
def compute_mt_ml(
    gt_data, track_data, min_track_percentage=0.8, max_track_percentage=0.2
):
    unique_ids = gt_data["id"].unique()
    mt_count = 0
    ml_count = 0

    for obj_id in unique_ids:
        gt_frames = set(gt_data[gt_data["id"] == obj_id]["frame"])
        track_frames = set(track_data[track_data["id"] == obj_id]["frame"])
        tracked_percentage = len(track_frames.intersection(gt_frames)) / len(gt_frames)

        if tracked_percentage >= min_track_percentage:
            mt_count += 1
        elif tracked_percentage <= max_track_percentage:
            ml_count += 1

    return mt_count, ml_count


# 3. Fragmentation
def compute_fragmentation(gt_data, track_data):
    fragments = 0
    for obj_id in gt_data["id"].unique():
        gt_frames = sorted(gt_data[gt_data["id"] == obj_id]["frame"].values)
        track_frames = sorted(track_data[track_data["id"] == obj_id]["frame"].values)
        fragments += len(set(gt_frames).difference(track_frames))

    return fragments


# 4. False Positives & False Negatives
def compute_fp_fn(gt_data, track_data):
    frames = sorted(gt_data["frame"].unique())
    fp, fn = 0, 0

    for frame in frames:
        gt_ids = gt_data[gt_data["frame"] == frame]["id"].values
        track_ids = track_data[track_data["frame"] == frame]["id"].values

        fn += len(
            set(gt_ids) - set(track_ids)
        )  # False negatives: missed ground truth IDs
        fp += len(
            set(track_ids) - set(gt_ids)
        )  # False positives: extra IDs detected by tracker

    return fp, fn


# 5. Tracking Precision & Recall
def compute_precision_recall(gt_data, track_data):
    tp = 0
    for frame in sorted(gt_data["frame"].unique()):
        gt_ids = gt_data[gt_data["frame"] == frame]["id"].values
        track_ids = track_data[track_data["frame"] == frame]["id"].values

        tp += len(set(gt_ids) & set(track_ids))  # True positives
    fp, fn = compute_fp_fn(gt_data, track_data)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    return precision, recall


# 6. MOTA
def compute_mota(gt_data, track_data):
    fp, fn = compute_fp_fn(gt_data, track_data)
    id_switches = compute_id_switches(gt_data, track_data)
    total_gt = len(gt_data)

    mota = 1 - (fp + fn + id_switches) / total_gt
    return mota


# 7. MOTP
def compute_motp(gt_data, track_data):
    total_iou = 0
    matches = 0

    for frame in sorted(gt_data["frame"].unique()):
        gt_boxes = gt_data[gt_data["frame"] == frame][["x1", "y1", "x2", "y2"]].values
        track_boxes = track_data[track_data["frame"] == frame][
            ["x1", "y1", "x2", "y2"]
        ].values

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
def compute_idf1(gt_data, track_data):
    precision, recall = compute_precision_recall(gt_data, track_data)
    idf1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )
    return idf1

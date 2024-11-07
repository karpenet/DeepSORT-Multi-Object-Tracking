from metrics import EvaluateMOTS

evaluation = EvaluateMOTS(
    gt_file_path="dataset/KITTI/labels/0005.txt",
    tracks_file_path="dataset/KITTI/eval/0005.txt",
)

metrics = evaluation.evaluate_deepsort()
print(metrics)
evaluation.update_metrics_csv(results_file_path="evaluate/results.csv")
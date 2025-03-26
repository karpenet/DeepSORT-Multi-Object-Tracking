from ultralytics import YOLO


model = YOLO("../yolo11s.pt").to('cuda')
model.export(
    format="engine",
    imgsz=640, 
    batch=1, 
    half=True, 
    device=0,
    data='../../dataset/KITTI/kitti.yaml',
)
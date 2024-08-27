from ultralytics import YOLO


model = YOLO("../yolov8n.pt").to('cuda')
model.export(format="engine", imgsz=128, batch=1, half=True)
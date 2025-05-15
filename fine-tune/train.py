from ultralytics import YOLO

# 載入預訓練模型
model = YOLO("yolov8n.pt")

# 開始訓練
model.train(data="/workspace/fine-tune/datasets/Granite-Getaway-2/data.yaml", epochs=400, imgsz=640)

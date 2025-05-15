from ultralytics import YOLO

# 載入預訓練模型（可選 yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt）
model = YOLO("yolov8m.pt")  # nano 版本，速度快、適合測試

# 預測圖片
results = model("images/test.png", save=True)  # 預測後自動存圖（在 runs/detect/predict 目錄）

# 顯示結果（可選）
results[0].show()

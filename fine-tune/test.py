from ultralytics import YOLO

# 1. 載入你訓練好的模型
model = YOLO("./runs/detect/train14/weights/best.pt")

# 2. 設定輸入影片路徑
video_path = "./test/Gameplay.mp4"  # 改成你自己的影片路徑

# 3. 執行推論
results = model(
    source=video_path,   # 輸入影片
    save=True,           # 儲存結果影片
    conf=0.4,            # 信心門檻（預設是 0.25，可微調）
    iou=0.5,             # NMS 門檻（可選）
    show=False,          # 設 True 可即時預覽（需有 GUI）
)

# 4. 推論完成後，影片會存到 runs/detect/predict/
print("✅ 推論完成！結果儲存在：runs/detect/predictX/")

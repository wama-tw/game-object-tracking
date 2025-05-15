import cv2
from ultralytics import YOLO
from yolox.tracker.byte_tracker import BYTETracker, STrack
import numpy as np
from types import SimpleNamespace
from datetime import datetime  # 新增 datetime 模組

CONF = 0.3

# 初始化 YOLO 模型
model = YOLO("runs/detect/train14/weights/best.pt")

# 初始化 ByteTrack 參數（用 SimpleNamespace 代替 dict）
args = SimpleNamespace(
    track_thresh=0.3,
    match_thresh=0.8,
    track_buffer=30,
    frame_rate=30,
    det_thresh=0.4,
    min_box_area=10,
    mot20=False
)
tracker = BYTETracker(args, frame_rate=args.frame_rate)

# 開啟影片
cap = cv2.VideoCapture("./test/Gameplay.mp4")
width, height = int(cap.get(3)), int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# 使用當前時間作為檔名
current_time = datetime.now().strftime("%Y%m%d%H%M")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(f"./output_tracked/{current_time}.mp4", fourcc, fps, (width, height))

frame_id = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1

    # YOLO 推論
    results = model.predict(source=frame, conf=CONF, verbose=False)
    detections = results[0].boxes

    # 整理 ByteTrack 格式：[x1, y1, x2, y2, score]
    if detections is not None and detections.cls.numel() > 0:
        dets = []
        for box in detections:
            xyxy = box.xyxy.cpu().numpy()[0]
            conf = box.conf.cpu().numpy()[0]
            dets.append([*xyxy, conf])
        online_targets = tracker.update(np.array(dets), (height, width), (height, width))
    else:
        online_targets = []

    # 繪製追蹤結果
    for track in online_targets:
        tlwh = track.tlwh
        tid = track.track_id
        x1, y1, x2, y2 = int(tlwh[0]), int(tlwh[1]), int(tlwh[0]+tlwh[2]), int(tlwh[1]+tlwh[3])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID {tid}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    out.write(frame)
    print(f'Processed frame {frame_id}')

cap.release()
out.release()
print(f"✅ 追蹤影片輸出完成：output_tracked/{current_time}.mp4")
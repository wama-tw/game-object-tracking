import cv2
from ultralytics import YOLO
from yolox.tracker.byte_tracker import BYTETracker
import numpy as np
from types import SimpleNamespace
from datetime import datetime

CONF = 0.4
current_time = datetime.now().strftime("%Y%m%d%H%M")
OUTPUT_PATH = f"./output_tracked/{current_time}-conf{CONF}.mp4"

# 指定要追蹤的類別
FIXED_IDS = {
    'Luigi': 1,
    'Boo': 2,
    'Peach': 3,
    'Shy Guy': 4
}
TARGET_CLASSES = set(FIXED_IDS.keys()) | {'obstacle'}

# 初始化 YOLO 模型
model = YOLO("runs/detect/train14/weights/best.pt")

# 初始化 ByteTrack（用於 obstacle）
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

# 輸入與輸出
cap = cv2.VideoCapture("./test/Gameplay.mp4")
width, height = int(cap.get(3)), int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

frame_id = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_id += 1

    results = model.predict(source=frame, conf=CONF, verbose=False)
    detections = results[0].boxes

    role_dets = []      # 固定 ID 類別
    obstacle_dets = []  # ByteTrack 類別

    if detections is not None and detections.cls.numel() > 0:
        for box in detections:
            cls_id = int(box.cls.cpu().numpy()[0])
            cls_name = model.names[cls_id]
            if cls_name not in TARGET_CLASSES:
                continue

            xyxy = box.xyxy.cpu().numpy()[0]
            conf = box.conf.cpu().numpy()[0]
            x1, y1, x2, y2 = map(int, xyxy)

            if cls_name in FIXED_IDS:
                role_dets.append((x1, y1, x2, y2, conf, cls_name))
            elif cls_name == 'obstacle':
                obstacle_dets.append([x1, y1, x2, y2, conf])

    # 畫角色（固定 ID）
    for x1, y1, x2, y2, conf, cls_name in role_dets:
        tid = FIXED_IDS[cls_name]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(frame, f'{cls_name} (ID {tid})', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # 用 ByteTrack 處理 obstacle
    if obstacle_dets:
        print(f"[{frame_id}] obstacle dets: {len(obstacle_dets)}")
        print("obstacle boxes:", obstacle_dets)
        online_targets = tracker.update(np.array(obstacle_dets), (height, width), (height, width))
        print(f"→ ByteTrack output: {len(online_targets)} targets")
        for track in online_targets:
            tlwh = track.tlwh
            tid = track.track_id
            x1, y1, x2, y2 = int(tlwh[0]), int(tlwh[1]), int(tlwh[0]+tlwh[2]), int(tlwh[1]+tlwh[3])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'obstacle (ID {tid})', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    out.write(frame)
    print(f'Processed frame {frame_id}')

cap.release()
out.release()
print(f"✅ 追蹤影片輸出完成：{OUTPUT_PATH}")

# 🎮 Mario Fixed-ID Tracker

This project provides a custom object tracking system using [YOLOv8](https://github.com/ultralytics/ultralytics) and [ByteTrack](https://github.com/ifzhang/ByteTrack) to analyze gameplay videos.  
It focuses on tracking **specific game characters with fixed IDs** — including Luigi, Boo, Peach, and Shy Guy — and dynamically tracks generic obstacles using ByteTrack.

---

## 🔧 Features

- 🎯 Fixed ID tracking for main characters:
  - `Luigi` → ID 1  
  - `Boo` → ID 2  
  - `Peach` → ID 3  
  - `Shy Guy` → ID 4
- 🟩 Dynamic object tracking for `'obstacle'` using ByteTrack
- ✅ Works with YOLOv8 fine-tuned custom models
- 🕹 Accurate for gameplay analysis, emotion interaction, or data collection

---

## 📁 Folder Structure

```

.
├── test/
│   └── Gameplay.mp4               # Input video
├── output\_tracked/
│   └── 20240515-conf0.4.mp4       # Output video with tracked boxes
├── runs/
│   └── detect/train14/weights/    # Your YOLOv8 trained model
├── yolov8\_fixedid\_bytetrack.py   # Main tracking script

```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install ultralytics
pip install lap cython_bbox filterpy
pip install git+https://github.com/ifzhang/ByteTrack.git
```

### 2. Place your fine-tuned YOLOv8 model at:

```
runs/detect/train14/weights/best.pt
```

### 3. Put your input video here:

```
./test/Gameplay.mp4
```

### 4. Run the tracker

```bash
python yolov8_fixedid_bytetrack.py
```

### 5. Output will be saved as:

```
./output_tracked/<timestamp>-conf<score>.mp4
```

---

## 🧠 Model Requirements

Make sure your YOLOv8 model was trained with the following class names in `data.yaml`:

```yaml
names:
  0: Luigi
  1: Boo
  2: Peach
  3: Shy Guy
  4: obstacle
```

---

## 📈 Example Output

![Tracking example](./docs/tracking_example.gif)  <!-- Optional: Replace with your own frame GIF or video -->

---

## 📤 Future Ideas

* [ ] CSV export of tracking logs (frame-by-frame position, ID)
* [ ] Invisible persistence (tracking through occlusion)
* [ ] Web UI to review tracking results
* [ ] Interactive labeling pipeline

---

## 📄 License

MIT License.
This repo is designed for research and non-commercial use in gameplay studies and interaction design.


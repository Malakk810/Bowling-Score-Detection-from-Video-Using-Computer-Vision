# Bowling Score Detection from Video Using Computer Vision


A CV pipeline that takes a single 30-second phone recording of seven bowling throws and produces an annotated output video that marks each pin fall with a timestamp, tracks a live pins-down counter, and ends with a summary screen. Ten pins are arranged in a rectangular layout; detection and tracking run on a YOLOv8n model fine-tuned on frames extracted from the source video.

## How it works

1. **Frame extraction and keyframe selection.** All 900 frames (30 fps, 720×650) are extracted with OpenCV, then a manually chosen set of frame ranges around each throw's impact/toppling/settling activity is kept, discarding static segments (ball approach, empty lane, pinsetter reset). This yields 412 keyframes out of 900.
2. **Annotation.** Keyframes are labeled in Roboflow with two classes, `Standing_Pin` and `Fallen_Pin`, and exported in YOLOv8 format (80/20 train/valid split, 412 images, ~1,203 total instances).
3. **Detection model.** YOLOv8n (COCO-pretrained, 3.0M params, 8.1 GFLOPs) is fine-tuned for 60 epochs (patience 15, image size 640, batch 16, AdamW, lr0=0.01, horizontal flip only, scale/translate augmentation, no rotation/vertical flip since the camera and pins are fixed-orientation).
4. **Tracking.** Ultralytics' built-in ByteTrack (`persist=True`) assigns a consistent ID to each detected pin across frames within a throw.
5. **Fall detection.** Both classes are treated as "pin present" — fall detection is disappearance-based rather than relying on the `Fallen_Pin` class score, because that class is less reliable (see Results). A pin must be tracked for at least `STABLE_FRAMES=8` frames before it's eligible to "fall," then absent for `GONE_FRAMES=6` consecutive frames before the fall is confirmed. The first `WARMUP_SEC=1.5` seconds of each throw are ignored to avoid startup false positives. A duplicate check rejects any fall within `FALL_PROXIMITY=35` px of an already-recorded fall, since ByteTrack assigns fresh IDs after each pinsetter reset and would otherwise double-count the same physical pin.
6. **Rendering.** Active pins get a green box; a fallen pin gets a red circle at its last known position for `MARKER_DURATION=2.0` seconds. A HUD shows elapsed time and cumulative pins down, followed by a 4-second summary screen.

## Results

Validation set (82 images, 670 instances):

| Metric | Overall | Standing_Pin | Fallen_Pin |
|---|---|---|---|
| mAP@0.50 | 0.862 | 0.920 | 0.803 |
| mAP@0.50:0.95 | 0.757 | 0.854 | 0.660 |
| Precision | 0.832 | 0.918 | 0.746 |
| Recall | 0.862 | 0.920 | 0.803 |

`Fallen_Pin` underperforms `Standing_Pin` due to a ~9:1 class imbalance and greater visual variability (partially occluded, lying flat, near frame edge) — the motivation for disappearance-based fall detection instead of trusting the `Fallen_Pin` confidence score directly.

On the full 30-second, 7-throw test recording: 101 unique tracking IDs (new IDs assigned after each pinsetter reset), 36 pins knocked down, first fall at 3.7 s (pin 7), last fall at 28.0 s (pin 118).

## Repository structure

*(Assumed — update to match your actual layout.)*

```
.
├── CV_Project_Bowling.ipynb   # end-to-end notebook: extraction, training, inference
├── DSAI352_BowlingReport.pdf  # written report
├── report.json                # per-pin fall events from the inference run
└── README.md
```

The notebook was written for Google Colab and expects the project directory mounted from Google Drive, with the source video at `CV_Project_Video.mov` / `CV_Project_Video_final.mp4` inside it.

## Running it

1. Install dependencies: `pip install roboflow ultralytics opencv-python`.
2. Point `PROJECT_DIR`/`DATASET_DIR`/`VIDEO_PATH` at your own paths (the notebook currently hardcodes a Google Drive mount).
3. Set `ROBOFLOW_API_KEY` as an environment variable (the notebook reads it via `os.environ.get("ROBOFLOW_API_KEY", "YOUR_API_KEY")` — don't hardcode it), and fill in your own Roboflow workspace/project names. Run the frame extraction and keyframe-selection cells, or skip straight to downloading the labeled dataset if you're reusing existing annotations.
4. Train: `model.train(data=data_yaml, epochs=60, imgsz=640, batch=16, optimizer='AdamW', lr0=0.01, ...)` — see the notebook for full arguments.
5. Run `process_video(...)` on the target `.mp4`/`.mov` to produce the annotated output video and `report.json`.

## Limitations

- Class imbalance holds `Fallen_Pin` recall down (0.803); more fallen-pin samples or class-weighted loss would help, though it wouldn't change the disappearance-based detector itself.
- ByteTrack reassigns IDs after each pinsetter reset (101 IDs tracked for 10 physical pins over 7 throws); the proximity-based duplicate check mitigates but doesn't eliminate double-counting risk.
- The fall marker uses the last known centroid before disappearance, so partial occlusion at that moment can shift it slightly.
- Assumes a fixed camera; significant hand movement during a throw could trigger false disappearances.
- No ball tracking — contact timing isn't used to confirm a fall is real versus an occlusion.

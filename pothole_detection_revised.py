#!/usr/bin/env python3
"""
pothole_detection_revised.py  (detection-only version)
=======================================================
Video in  ->  annotated video + Excel/CSV dataset + snapshots out.

    python pothole_detection_revised.py road.mp4
    python pothole_detection_revised.py road.mp4 --classes pothole,patching
    python pothole_detection_revised.py road.mp4 --classes all --conf 0.25
    python pothole_detection_revised.py --list-classes

WHAT THIS VERSION DOES
    1. Runs the 8-class YOLO11n NCNN model on every Nth frame.
    2. Tracks each defect across frames (greedy IoU tracker, class-aware).
    3. Counts a defect only if it was seen in >= --min-sightings frames
       (removes one-frame flicker).
    4. Writes the annotated video, potholes.xlsx (or CSV), and a cropped
       snapshot of the best view of every defect.

REMOVED (compared with the previous "revised" version)
    - dynamic pitch/roll estimation (TiltEstimator)
    - visual odometry + heading Kalman filter
    - GPS loading/interpolation, lat/lon <-> local-metric conversion
    - pixel-to-ground projection, multi-frame triangulation, position
      de-duplication, and the HTML map
    - --gps/--lat/--lon/--heading/--cam-*/--hfov/--max-range/--merge-radius

CONSEQUENCES OF DROPPING LOCATION (read this)
    - There is no position, so nothing can be merged by "being in the same
      place". A defect counts once per TRACK. If the same pothole disappears
      for longer than --max-gap processed frames (occlusion, leaving the
      frame and re-entering, detector flicker) it gets a new track and is
      counted again. Treat the total as an upper bound.
    - There are no metres. Size is reported in pixels and as % of the frame;
      the old width-in-metres "severity" label is gone.

FUNCTIONS
    resolve_model_path, load_model, get_class_names, get_model_imgsz,
    parse_target_classes, detect_frame           detection
    SimpleTracker                                same defect across frames
    summarize_tracks                             tracks -> one row per defect
    annotate_frame                               draw on video
    save_snapshots, save_to_excel                outputs
    run_pipeline, main                           glue
"""
import argparse
import csv
import math
import os
import sys
import time
from collections import defaultdict

import cv2

# ============================================================================
# USER SETTINGS -- edit these, or override on the command line
# ============================================================================
DEFAULT_MODEL = "models/best_ncnn_model"
DEFAULT_TARGET_CLASSES = "pothole"      # names, ids, comma list, or "all"
CONF_THRESHOLD = 0.35
MIN_SIGHTINGS = 3                       # processed frames a defect must appear in
TRACK_IOU = 0.15                        # min box overlap to continue a track
TRACK_MAX_GAP = 5                       # processed frames a track may vanish for

# Fallback only -- normally names come from the model's own metadata.yaml.
DEFAULT_CLASS_NAMES = {
    0: "longitudinal_transverse_crack", 1: "alligator_crack", 2: "block_crack",
    3: "patching", 4: "pothole", 5: "weathering_raveling",
    6: "lane_shoulder_drop_off", 7: "marking_manhole",
}
SHORT_NAMES = {   # keeps on-video labels readable
    "longitudinal_transverse_crack": "long/trans crack", "alligator_crack": "alligator",
    "block_crack": "block crack", "patching": "patch", "pothole": "pothole",
    "weathering_raveling": "raveling", "lane_shoulder_drop_off": "drop-off",
    "marking_manhole": "manhole",
}
CLASS_COLORS_BGR = [(0, 165, 255), (255, 120, 0), (200, 0, 200), (0, 200, 200),
                    (0, 60, 255), (120, 200, 0), (255, 200, 0), (150, 150, 150)]


# ============================================================================
# 1. DETECTION
# ============================================================================
def resolve_model_path(path):
    """Validate the model path and fail with something actionable. An NCNN
    export is a FOLDER containing model.ncnn.param / model.ncnn.bin (and the
    metadata.yaml Ultralytics needs for class names + task)."""
    candidates = [path, os.path.join("models", path)]
    for c in candidates:
        if os.path.isfile(c):
            return c
        if os.path.isdir(c):
            if not any(f.endswith(".param") for f in os.listdir(c)):
                sys.exit(f"'{c}' is a folder but has no .param file -- is it a complete "
                         f"NCNN export? (expected model.ncnn.param + model.ncnn.bin)")
            return c
    found = []
    for root in (".", "models"):
        if os.path.isdir(root):
            for name in sorted(os.listdir(root)):
                full = os.path.join(root, name)
                if name.endswith(".pt") or (os.path.isdir(full) and "ncnn" in name.lower()):
                    found.append(full)
    hint = ("\n  found nearby: " + ", ".join(found)) if found else ""
    sys.exit(f"model not found: '{path}' (also tried 'models/{path}').{hint}")


def load_model(path):
    """Load a YOLO model: a .pt file or an NCNN export folder. The model is
    detect-only, so task is pinned to 'detect'."""
    from ultralytics import YOLO
    return YOLO(resolve_model_path(path), task="detect")


def _read_metadata(path):
    """metadata.yaml of an NCNN export folder, or {} if unavailable."""
    if path and os.path.isdir(path):
        meta = os.path.join(path, "metadata.yaml")
        if os.path.isfile(meta):
            try:
                import yaml
                with open(meta, encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
            except Exception:
                pass
    return {}


def get_class_names(model, path=None):
    """{id: name}. For an NCNN folder this reads metadata.yaml directly: asking
    an exported Ultralytics model for .names makes it load the whole network
    once just to answer, and then AGAIN on the first predict()."""
    names = _read_metadata(path).get("names")
    if not names:
        names = getattr(model, "names", None)
    if isinstance(names, dict) and names:
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, (list, tuple)) and names:
        return {i: str(v) for i, v in enumerate(names)}
    return dict(DEFAULT_CLASS_NAMES)


def get_model_imgsz(path, fallback=640):
    """Input size the NCNN graph was exported with ([h, w] or an int). The
    graph is fixed-shape, so read it from the export instead of guessing."""
    sz = _read_metadata(path).get("imgsz")
    if isinstance(sz, (list, tuple)) and len(sz) == 2:
        return int(sz[0]) if sz[0] == sz[1] else [int(sz[0]), int(sz[1])]
    if isinstance(sz, int):
        return sz
    return fallback


def _padding_hint(W, H, imgsz):
    """If the video is far from square but the model input is square, a large
    part of every processed frame is grey letterbox padding (wasted compute)."""
    if not isinstance(imgsz, int) or abs(W - H) / max(W, H) < 0.25:
        return None
    scale = imgsz / max(W, H)
    short = int(math.ceil(min(W, H) * scale / 32.0) * 32)
    waste = 100.0 * (1.0 - min(W, H) * scale / imgsz)
    shape = f"{imgsz},{short}" if H >= W else f"{short},{imgsz}"
    return (f"NOTE: video is {W}x{H} but the model input is {imgsz}x{imgsz}: ~{waste:.0f}% of "
            f"every frame the network processes is grey padding. A rectangular export is "
            f"~{100 * (1 - short / imgsz):.0f}% cheaper at identical accuracy:\n"
            f"      yolo export model=models/best.pt format=ncnn imgsz={shape}")


def parse_target_classes(spec, names):
    """'pothole' | '4' | 'pothole,patching' | 'all'  ->  sorted list of class ids."""
    spec = (spec or "").strip().lower()
    if spec in ("all", "*"):
        return sorted(names)
    by_name = {v.lower(): k for k, v in names.items()}
    ids = []
    for tok in [t.strip() for t in spec.split(",") if t.strip()]:
        if tok.isdigit() and int(tok) in names:
            ids.append(int(tok))
        elif tok in by_name:
            ids.append(by_name[tok])
        else:
            sys.exit(f"unknown class '{tok}'. Available: "
                     + ", ".join(f"{k}={v}" for k, v in sorted(names.items())))
    if not ids:
        sys.exit("--classes is empty")
    return sorted(set(ids))


def detect_frame(model, frame, conf=CONF_THRESHOLD, imgsz=640, device=None, classes=None):
    """Run the model on one BGR frame. Returns
    [{x1,y1,x2,y2,conf,cls}, ...] in original-frame pixels. `classes` is a
    list of class ids to keep (filtered inside Ultralytics' NMS); None = all."""
    res = model.predict(frame, imgsz=imgsz, conf=conf, device=device,
                        classes=classes, verbose=False)[0]
    dets = []
    if res.boxes is None:
        return dets
    xyxy = res.boxes.xyxy.cpu().numpy()
    confs = res.boxes.conf.cpu().numpy()
    clss = res.boxes.cls.cpu().numpy().astype(int)
    for b, c, k in zip(xyxy, confs, clss):
        dets.append({"x1": float(b[0]), "y1": float(b[1]),
                     "x2": float(b[2]), "y2": float(b[3]),
                     "conf": float(c), "cls": int(k)})
    return dets


# ============================================================================
# 2. TRACKING (same defect across consecutive frames)
# ============================================================================
def _iou(a, b):
    ix = max(0.0, min(a["x2"], b["x2"]) - max(a["x1"], b["x1"]))
    iy = max(0.0, min(a["y2"], b["y2"]) - max(a["y1"], b["y1"]))
    inter = ix * iy
    ua = ((a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
          + (b["x2"] - b["x1"]) * (b["y2"] - b["y1"]) - inter)
    return inter / ua if ua > 0 else 0.0


class SimpleTracker:
    """Greedy IoU tracker. Gives each detection a `track_id`; boxes that overlap
    a recent box of the SAME CLASS get the same id. `frame_no` counts
    PROCESSED frames."""

    def __init__(self, iou_thr=TRACK_IOU, max_gap=TRACK_MAX_GAP):
        self.iou_thr, self.max_gap = iou_thr, max_gap
        self.tracks = {}
        self.next_id = 1

    def update(self, dets, frame_no):
        pairs = []
        for di, d in enumerate(dets):
            for tid, tr in self.tracks.items():
                if frame_no - tr["last"] > self.max_gap:
                    continue
                if d.get("cls") != tr["box"].get("cls"):
                    continue          # never link e.g. a crack to a pothole
                v = _iou(d, tr["box"])
                if v >= self.iou_thr:
                    pairs.append((v, di, tid))
        pairs.sort(reverse=True)
        used_d, used_t = set(), set()
        for v, di, tid in pairs:
            if di in used_d or tid in used_t:
                continue
            dets[di]["track_id"] = tid
            self.tracks[tid] = {"box": dets[di], "last": frame_no}
            used_d.add(di)
            used_t.add(tid)
        for di, d in enumerate(dets):
            if di not in used_d:
                d["track_id"] = self.next_id
                self.tracks[self.next_id] = {"box": d, "last": frame_no}
                self.next_id += 1
        self.tracks = {k: v for k, v in self.tracks.items()
                       if frame_no - v["last"] <= self.max_gap}
        return dets


# ============================================================================
# 3. TRACKS -> ONE ROW PER DEFECT
# ============================================================================
def summarize_tracks(track_groups, class_names, min_sightings=MIN_SIGHTINGS):
    """track_groups: {track_id: [observation dict, ...]}. Keeps tracks seen at
    least `min_sightings` times and returns one summary row per track, ordered
    by first appearance. No position is involved, so nothing is merged."""
    rows = []
    for tid, obs in track_groups.items():
        if len(obs) < min_sightings:
            continue
        best = max(obs, key=lambda o: o["conf"])
        cls = obs[0]["cls"]
        rows.append({
            "track_id": tid,
            "cls": cls,
            "class_name": class_names.get(cls, str(cls)),
            "sightings": len(obs),
            "best_conf": best["conf"],
            "mean_conf": sum(o["conf"] for o in obs) / len(obs),
            "max_box_area_pct": max(o["area_pct"] for o in obs),
            "first_seen_s": min(o["t"] for o in obs),
            "last_seen_s": max(o["t"] for o in obs),
            "best_frame": best["frame"],
            "best_box": (best["x1"], best["y1"], best["x2"], best["y2"]),
            "snapshot": "",
        })
    rows.sort(key=lambda r: r["first_seen_s"])
    for i, r in enumerate(rows, 1):
        r["pothole_id"] = i
    return rows


# ============================================================================
# 4. VIDEO OVERLAY
# ============================================================================
def annotate_frame(frame, dets, n_tracks_so_far, fps_text="", class_names=None):
    class_names = class_names or DEFAULT_CLASS_NAMES
    for d in dets:
        p1, p2 = (int(d["x1"]), int(d["y1"])), (int(d["x2"]), int(d["y2"]))
        k = d.get("cls", 0)
        color = CLASS_COLORS_BGR[k % len(CLASS_COLORS_BGR)]
        cv2.rectangle(frame, p1, p2, color, 3)
        full = class_names.get(k, str(k))
        label = f"#{d['track_id']} {SHORT_NAMES.get(full, full)} {d['conf'] * 100:.0f}%"
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        y0 = max(0, p1[1] - th - bl)
        cv2.rectangle(frame, (p1[0], y0), (p1[0] + tw, y0 + th + bl), color, -1)
        cv2.putText(frame, label, (p1[0], y0 + th), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"in frame: {len(dets)}   tracks so far: {n_tracks_so_far}  {fps_text}",
                (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return frame


# ============================================================================
# 5. OUTPUTS
# ============================================================================
def save_snapshots(video_path, defects, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    for p in defects:
        cap.set(cv2.CAP_PROP_POS_FRAMES, p["best_frame"])
        ok, frame = cap.read()
        if not ok:
            continue
        x1, y1, x2, y2 = p["best_box"]
        mx, my = (x2 - x1) * 0.4, (y2 - y1) * 0.4
        h, w = frame.shape[:2]
        crop = frame[max(0, int(y1 - my)):min(h, int(y2 + my)),
                     max(0, int(x1 - mx)):min(w, int(x2 + mx))]
        if crop.size == 0:
            continue
        path = os.path.join(out_dir, f"{p['class_name']}_{p['pothole_id']:03d}.jpg")
        cv2.imwrite(path, crop)
        p["snapshot"] = path
    cap.release()


def _write_csv(path, rows, cols):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow(["" if r.get(c) is None else r.get(c) for c in cols])


def save_to_excel(observations, defects, settings, path):
    """Write potholes.xlsx (Potholes / Detections / Settings). If pandas or
    openpyxl is missing, writes plain CSV files instead (stdlib only, so this
    step can never crash the run) and tells you what to install."""
    p_cols = ["pothole_id", "class_name", "sightings", "best_conf", "mean_conf",
              "max_box_area_pct", "first_seen_s", "last_seen_s", "snapshot"]
    d_cols = ["frame", "t", "track_id", "class_name", "conf",
              "x1", "y1", "x2", "y2", "area_pct"]
    try:
        import pandas as pd
        df_p = pd.DataFrame(defects, columns=p_cols)
        df_d = pd.DataFrame(observations, columns=d_cols)
        df_s = pd.DataFrame(list(settings.items()), columns=["setting", "value"])
        with pd.ExcelWriter(path, engine="openpyxl") as xw:
            df_p.to_excel(xw, sheet_name="Potholes", index=False)
            df_d.to_excel(xw, sheet_name="Detections", index=False)
            df_s.to_excel(xw, sheet_name="Settings", index=False)
        return path
    except ImportError as e:
        stem = os.path.splitext(path)[0]
        print(f"NOTE: Excel export unavailable ({e}); writing CSV files instead. "
              f"Fix with:  pip install pandas openpyxl")
        _write_csv(stem + "_potholes.csv", defects, p_cols)
        _write_csv(stem + "_detections.csv", observations, d_cols)
        _write_csv(stem + "_settings.csv",
                   [{"setting": k, "value": v} for k, v in settings.items()],
                   ["setting", "value"])
        return stem + "_potholes.csv"


# ============================================================================
# 6. MAIN PIPELINE
# ============================================================================
def run_pipeline(args, detector=None):
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"cannot open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if detector is None:
        model = load_model(args.model)
        model_path = resolve_model_path(args.model)
        class_names = get_class_names(model, model_path)
        if args.imgsz is None:
            args.imgsz = get_model_imgsz(model_path)
        hint = _padding_hint(W, H, args.imgsz)
        if hint:
            print(hint)
        target_ids = parse_target_classes(args.classes, class_names)
        # pass None (no filter) when every class is selected
        cls_filter = None if len(target_ids) == len(class_names) else target_ids
        detector = lambda fr: detect_frame(model, fr, args.conf, args.imgsz,
                                           args.device, cls_filter)
    else:
        # injected detector (tests / other backends): names supplied via args
        class_names = getattr(args, "class_names", None) or dict(DEFAULT_CLASS_NAMES)
        target_ids = parse_target_classes(args.classes, class_names)
        if args.imgsz is None:
            args.imgsz = 640
    print("classes tracked : " + ", ".join(f"{k}={class_names[k]}" for k in target_ids))

    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.video))[0]
    video_out = os.path.join(args.out_dir, f"{stem}_detected.mp4")
    writer = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps / args.stride, (W, H))

    tracker = SimpleTracker(max_gap=args.max_gap)
    track_groups = {}
    observations = []
    frame_area = float(W * H)
    frame_idx, processed = -1, 0
    timers = defaultdict(float)     # seconds per stage, for the end-of-run summary
    pc = time.perf_counter
    t_start = time.time()

    print("running -- Ctrl+C to stop early")

    while True:
        tk = pc()
        if not cap.grab():
            break
        frame_idx += 1
        if frame_idx % args.stride:
            timers["video decode"] += pc() - tk
            continue
        ok, frame = cap.retrieve()
        timers["video decode"] += pc() - tk
        if not ok:
            break
        t_video = frame_idx / fps

        tk = pc()
        dets = detector(frame)
        timers["detection (model)"] += pc() - tk

        tk = pc()
        # belt-and-braces: keeps injected detectors honest, no-op for the YOLO path
        dets = [d for d in dets if d.get("cls", target_ids[0]) in target_ids]
        for d in dets:
            d.setdefault("cls", target_ids[0])
        tracker.update(dets, processed)
        for d in dets:
            area_pct = 100.0 * (d["x2"] - d["x1"]) * (d["y2"] - d["y1"]) / frame_area
            row = {"frame": frame_idx, "t": t_video, "track_id": d["track_id"],
                   "cls": d["cls"], "class_name": class_names.get(d["cls"], str(d["cls"])),
                   "conf": d["conf"], "x1": d["x1"], "y1": d["y1"],
                   "x2": d["x2"], "y2": d["y2"], "area_pct": area_pct}
            observations.append(row)
            track_groups.setdefault(d["track_id"], []).append(row)
        timers["tracking"] += pc() - tk

        tk = pc()
        rate = processed / max(time.time() - t_start, 1e-6)
        annotate_frame(frame, dets, tracker.next_id - 1, f"{rate:.1f} proc-fps", class_names)
        writer.write(frame)
        timers["draw + video write"] += pc() - tk

        if args.show:
            cv2.imshow("potholes", cv2.resize(frame, (max(W // 2, 1), max(H // 2, 1))))
            if cv2.waitKey(1) in (27, ord("q")):
                break
        processed += 1
        if processed % 50 == 0:
            print(f"  frame {frame_idx}/{total}  ({rate:.1f} frames/s processed)")

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    defects = summarize_tracks(track_groups, class_names, args.min_sightings)
    save_snapshots(args.video, defects, os.path.join(args.out_dir, "snapshots"))

    settings = {"video": args.video, "model": args.model,
                "classes": ", ".join(class_names[k] for k in target_ids),
                "conf": args.conf, "imgsz": str(args.imgsz), "stride": args.stride,
                "frame_size": f"{W}x{H}", "min_sightings": args.min_sightings,
                "track_max_gap": args.max_gap,
                "note": "no location: counts are per track, upper bound"}
    xlsx = save_to_excel(observations, defects, settings,
                         os.path.join(args.out_dir, "potholes.xlsx"))

    wall = max(time.time() - t_start, 1e-6)
    counts = defaultdict(int)
    for d_ in defects:
        counts[d_["class_name"]] += 1
    print("\nDone.")
    if defects:
        print(f"  unique defects  : {len(defects)}   ("
              + ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())) + ")")
    else:
        print("  unique defects  : 0")
    if processed:
        print(f"\n  TIMING  ({processed} processed frames, {wall:.1f}s wall, "
              f"{processed / wall:.2f} frames/s overall)")
        tsum = max(sum(timers.values()), 1e-9)
        for name, sec in sorted(timers.items(), key=lambda kv: -kv[1]):
            print(f"    {name:24s} {1000 * sec / processed:7.1f} ms/frame   "
                  f"{100 * sec / tsum:4.0f}%")
        print()
    print(f"  video           : {video_out}")
    print(f"  dataset         : {xlsx}")
    print(f"  snapshots       : {os.path.join(args.out_dir, 'snapshots')}")
    return defects


def main():
    ap = argparse.ArgumentParser(description="Pothole video -> dataset (detection + tracking only)")
    ap.add_argument("video", nargs="?", help="input video (not needed with --list-classes)")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="best.pt or an NCNN export folder (default: %(default)s)")
    ap.add_argument("--classes", default=DEFAULT_TARGET_CLASSES,
                    help='what to track: names/ids, comma-separated, or "all" '
                         '(default: %(default)s). See --list-classes.')
    ap.add_argument("--list-classes", action="store_true",
                    help="print the model's class ids/names and exit")
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--conf", type=float, default=CONF_THRESHOLD)
    ap.add_argument("--imgsz", type=int, nargs="+", default=None, metavar="N",
                    help="model input size: one number (square) or 'H W'. Default: read from "
                         "the NCNN export's metadata.yaml, which is what the graph expects.")
    ap.add_argument("--device", default=None,
                    help="ultralytics device string, e.g. 'cpu', '0' (default: auto)")
    ap.add_argument("--stride", type=int, default=2, help="process every Nth frame")
    ap.add_argument("--show", action="store_true", help="live preview window (q to stop)")
    ap.add_argument("--min-sightings", type=int, default=MIN_SIGHTINGS,
                    help="frames a defect must appear in to be counted")
    ap.add_argument("--max-gap", type=int, default=TRACK_MAX_GAP,
                    help="processed frames a track may vanish before a new id is issued")
    args = ap.parse_args()
    if args.imgsz is not None:
        args.imgsz = args.imgsz[0] if len(args.imgsz) == 1 else list(args.imgsz)
    if args.list_classes:
        names = get_class_names(load_model(args.model), resolve_model_path(args.model))
        print("classes in this model:")
        for k, v in sorted(names.items()):
            print(f"  {k} = {v}")
        return
    if not args.video:
        ap.error("the following arguments are required: video")
    run_pipeline(args)


if __name__ == "__main__":
    main()
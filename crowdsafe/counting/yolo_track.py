"""Person / head detectors returning ``supervision.Detections`` (architecture.md §4 Detector).

- :class:`YoloDetector` — Ultralytics YOLO (e.g. YOLO11l COCO), optional tiling with
  ``sv.InferenceSlicer`` for small heads, optional ByteTrack tracking for video.
- :class:`PersonHeadOnnx` — the CrowdHuman YOLO11s person+head ONNX model (``Sharath33/Person``),
  run with onnxruntime using its model-card preprocessing.
- :func:`reliability_features` — the per-zone/frame features used by the fusion (architecture.md §2.2).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import supervision as sv


class YoloDetector:
    """Ultralytics YOLO wrapper: BGR frame in, person detections out."""

    def __init__(self, weights: str, imgsz: int = 1280, conf: float = 0.25, iou: float = 0.7,
                 classes: list[int] | None = None, device: str | int | None = None,
                 slice_wh: int | None = None, overlap_px: int = 100, track: bool = False,
                 tracker_cfg: str = "bytetrack.yaml") -> None:
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.kw = dict(imgsz=imgsz, conf=conf, iou=iou, classes=classes if classes is not None else [0],
                       device=device, verbose=False)
        self.track, self.tracker_cfg = track, tracker_cfg
        self.slicer = None
        if slice_wh:
            if track:
                raise ValueError("Tiling and tracking together are not supported; track on full frames.")
            self.slicer = sv.InferenceSlicer(callback=self._predict, slice_wh=slice_wh, overlap_wh=overlap_px)

    def _predict(self, bgr: np.ndarray) -> sv.Detections:
        return sv.Detections.from_ultralytics(self.model.predict(bgr, **self.kw)[0])

    def __call__(self, bgr: np.ndarray) -> sv.Detections:
        """Detect (and optionally track) persons; tracker_id is set only when track=True."""
        if self.slicer is not None:
            return self.slicer(bgr)
        if self.track:
            res = self.model.track(bgr, persist=True, tracker=self.tracker_cfg, **self.kw)[0]
            return sv.Detections.from_ultralytics(res)
        return self._predict(bgr)


def letterbox(bgr: np.ndarray, h_in: int, w_in: int, pad_value: int = 114) -> tuple[np.ndarray, float, int, int]:
    """Aspect-preserving resize into an h_in×w_in canvas. Returns canvas, scale, left pad, top pad."""
    import cv2

    h, w = bgr.shape[:2]
    s = min(w_in / w, h_in / h)
    nh, nw = int(round(h * s)), int(round(w * s))
    canvas = np.full((h_in, w_in, 3), pad_value, np.uint8)
    top, left = (h_in - nh) // 2, (w_in - nw) // 2
    canvas[top:top + nh, left:left + nw] = cv2.resize(bgr, (nw, nh))
    return canvas, s, left, top


def decode_yolo(preds: np.ndarray, conf: float, iou: float, scale: float, left: int, top: int) -> sv.Detections:
    """Decode a raw YOLO output (4 + n_classes, anchors) to NMS'd detections in original-image pixels."""
    import cv2

    p = preds.T                                            # (anchors, 4 + n_classes)
    scores = p[:, 4:]
    cls, score = scores.argmax(1), scores.max(1)
    keep = score >= conf
    p, cls, score = p[keep], cls[keep], score[keep]
    if not len(p):
        return sv.Detections.empty()
    xyxy = np.column_stack([p[:, 0] - p[:, 2] / 2, p[:, 1] - p[:, 3] / 2,
                            p[:, 0] + p[:, 2] / 2, p[:, 1] + p[:, 3] / 2])
    xyxy = (xyxy - [left, top, left, top]) / scale
    xywh = np.column_stack([xyxy[:, :2], xyxy[:, 2:] - xyxy[:, :2]])
    idx = np.array(cv2.dnn.NMSBoxesBatched(xywh.tolist(), score.tolist(), cls.tolist(), conf, iou)).reshape(-1)
    return sv.Detections(xyxy=xyxy[idx].astype(np.float32), confidence=score[idx].astype(np.float32),
                         class_id=cls[idx].astype(int))


def decode_split(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, conf: float, iou: float,
                 scale: float, left: int, top: int) -> sv.Detections:
    """Decode an end-to-end style export with separate outputs: boxes (anchors, 4 as cx,cy,w,h),
    scores (anchors, 1) and class ids (anchors, 1). Same result format as :func:`decode_yolo`."""
    score, cls = scores.reshape(-1), classes.reshape(-1).astype(int)
    preds = np.zeros((4 + int(cls.max(initial=0)) + 1, len(score)), np.float32)
    preds[:4] = boxes.reshape(-1, 4).T
    preds[4 + cls, np.arange(len(score))] = score
    return decode_yolo(preds, conf, iou, scale, left, top)


class PersonHeadOnnx:
    """CrowdHuman person+head YOLO11s (ONNX). class_id 0 = person, 1 = head."""

    PERSON, HEAD = 0, 1

    def __init__(self, onnx_path: str | Path, conf: float = 0.2, iou: float = 0.6,
                 providers: list[str] | None = None) -> None:
        import os

        import onnxruntime as ort

        so = ort.SessionOptions()
        if os.environ.get("OMP_NUM_THREADS"):     # respect the container CPU quota (see crowdsafe.nbenv)
            so.intra_op_num_threads = int(os.environ["OMP_NUM_THREADS"])
        self.sess = ort.InferenceSession(str(onnx_path), sess_options=so, providers=providers or ["CPUExecutionProvider"])
        inp = self.sess.get_inputs()[0]
        self.name = inp.name
        h, w = inp.shape[2], inp.shape[3]
        self.h_in, self.w_in = (h, w) if isinstance(h, int) and isinstance(w, int) else (640, 640)
        self.conf, self.iou = conf, iou

    def __call__(self, bgr: np.ndarray) -> sv.Detections:
        """Detect persons and heads in a BGR frame (model card: BGR, /255, letterbox 114)."""
        canvas, s, left, top = letterbox(bgr, self.h_in, self.w_in)
        x = canvas.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        outs = self.sess.run(None, {self.name: x})
        if len(outs) >= 3:        # Sharath33/Person export: boxes, scores, classes
            return decode_split(outs[0][0], outs[1][0], outs[2][0], self.conf, self.iou, s, left, top)
        return decode_yolo(outs[0][0], self.conf, self.iou, s, left, top)


def reliability_features(dets: sv.Detections, overlap_iou: float = 0.3) -> dict[str, float]:
    """Fusion inputs for one zone/frame: count, mean confidence, share of overlapping boxes, mean box height."""
    n = len(dets)
    if n == 0:
        return {"n_det": 0, "mean_conf": 0.0, "overlap_share": 0.0, "mean_box_h": 0.0}
    iou = sv.box_iou_batch(dets.xyxy, dets.xyxy)
    np.fill_diagonal(iou, 0)
    conf = dets.confidence if dets.confidence is not None else np.ones(n)
    return {"n_det": n, "mean_conf": float(np.mean(conf)),
            "overlap_share": float((iou.max(axis=1) > overlap_iou).mean()),
            "mean_box_h": float(np.mean(dets.xyxy[:, 3] - dets.xyxy[:, 1]))}

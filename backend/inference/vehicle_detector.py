import os
import cv2
import numpy as np
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
from collections import Counter
from typing import Union, List, Dict, Any, Optional, Tuple
from ultralytics import YOLO
import supervision as sv

# Project directory paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(CURRENT_DIR)
ROOT_DIR = os.path.dirname(BACKEND_DIR)

# Target vehicle classes in COCO dataset:
# Class 2: car
# Class 3: motorcycle
# Class 5: bus
# Class 7: truck
TARGET_VEHICLE_CLASSES = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck",
}

class VehicleDetector:
    """
    Real Vehicle Detector for UrbanEye Edge AI using YOLO11n.
    Detects strictly: Car, Motorcycle, Bus, and Truck.
    Returns class, confidence, and bounding box.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = self._resolve_model_path(model_path)
        print(f"[VEHICLE_DETECTOR] Loading YOLO11n model from: {self.model_path}")
        self.model = YOLO(self.model_path)
        self.target_classes = TARGET_VEHICLE_CLASSES

    def _resolve_model_path(self, model_path: Optional[str]) -> str:
        if model_path and os.path.exists(model_path):
            return model_path

        # Check candidate locations in order
        candidates = [
            os.getenv("YOLO_MODEL_PATH"),
            os.path.join(BACKEND_DIR, "models", "yolo11n.pt"),
            os.path.join(ROOT_DIR, "yolo11n.pt"),
            os.path.join(BACKEND_DIR, "yolo11n.pt"),
            "yolo11n.pt",
        ]

        for cand in candidates:
            if cand and os.path.exists(cand):
                return cand

        # Fallback to model name
        return "yolo11n.pt"

    def detect(
        self,
        image_input: Union[str, np.ndarray],
        confidence_threshold: float = 0.25
    ) -> Dict[str, Any]:
        """
        Run real vehicle detection on a single image (file path or numpy BGR array).

        Args:
            image_input: File path to image or numpy ndarray (cv2 image).
            confidence_threshold: Minimum confidence score to retain detection (0.0 to 1.0).

        Returns:
            Dictionary containing:
            - success: bool
            - count: int (number of detected target vehicles)
            - detections: list of dicts with class, confidence, confidence_pct, and bbox
            - summary: dict with count per vehicle category
        """
        try:
            if isinstance(image_input, str):
                if not os.path.exists(image_input):
                    return {
                        "success": False,
                        "error": f"Image path not found: {image_input}",
                        "count": 0,
                        "detections": [],
                        "summary": {c: 0 for c in self.target_classes.values()},
                    }
                frame = cv2.imread(image_input)
            elif isinstance(image_input, np.ndarray):
                frame = image_input
            else:
                return {
                    "success": False,
                    "error": f"Unsupported image input type: {type(image_input)}",
                    "count": 0,
                    "detections": [],
                    "summary": {c: 0 for c in self.target_classes.values()},
                }

            if frame is None or frame.size == 0:
                return {
                    "success": False,
                    "error": "Empty or invalid image frame provided.",
                    "count": 0,
                    "detections": [],
                    "summary": {c: 0 for c in self.target_classes.values()},
                }

            h, w = frame.shape[:2]

            # Run YOLO11n inference
            results = self.model(frame, conf=confidence_threshold, verbose=False)

            detections: List[Dict[str, Any]] = []
            summary = {c: 0 for c in self.target_classes.values()}

            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                for box in boxes:
                    cls_id = int(box.cls[0].item())

                    # Strictly filter for Car, Motorcycle, Bus, Truck only
                    if cls_id not in self.target_classes:
                        continue

                    class_name = self.target_classes[cls_id]
                    conf = float(box.conf[0].item())
                    xyxy = box.xyxy[0].tolist()

                    x_min = max(0.0, min(float(w), float(xyxy[0])))
                    y_min = max(0.0, min(float(h), float(xyxy[1])))
                    x_max = max(0.0, min(float(w), float(xyxy[2])))
                    y_max = max(0.0, min(float(h), float(xyxy[3])))

                    box_w = max(0.0, x_max - x_min)
                    box_h = max(0.0, y_max - y_min)
                    cx = x_min + box_w / 2.0
                    cy = y_min + box_h / 2.0

                    detections.append({
                        "class": class_name,
                        "label": class_name,
                        "confidence": round(conf, 4),
                        "confidence_pct": round(conf * 100.0, 1),
                        "type": "vehicle",
                        "bbox": {
                            "x_min": round(x_min, 2),
                            "y_min": round(y_min, 2),
                            "x_max": round(x_max, 2),
                            "y_max": round(y_max, 2),
                            "x": round(cx, 2),
                            "y": round(cy, 2),
                            "width": round(box_w, 2),
                            "height": round(box_h, 2),
                            "box_2d": [round(x_min, 2), round(y_min, 2), round(x_max, 2), round(y_max, 2)],
                        },
                    })

                    summary[class_name] += 1

            # Sort detections by confidence descending
            detections.sort(key=lambda d: d["confidence"], reverse=True)
            summary["total"] = len(detections)

            return {
                "success": True,
                "count": len(detections),
                "detections": detections,
                "summary": summary,
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "count": 0,
                "detections": [],
                "summary": {c: 0 for c in self.target_classes.values()},
            }

    def detect_frames(
        self,
        frames: List[Tuple[int, np.ndarray]],
        confidence_threshold: float = 0.25
    ) -> Dict[str, Any]:
        """
        Run vehicle detection on a sequence of sampled video frames.

        Args:
            frames: List of (frame_index, frame_image) tuples.
            confidence_threshold: Minimum confidence threshold.

        Returns:
            Dict containing per-frame detections, video-level summary, and max vehicle frame.
        """
        per_frame_results: Dict[int, List[Dict[str, Any]]] = {}
        total_detections_across_video = 0
        overall_summary = {c: 0 for c in self.target_classes.values()}

        max_vehicles_in_single_frame = 0
        best_vehicle_frame_idx: Optional[int] = None
        best_vehicle_detections: List[Dict[str, Any]] = []

        for f_idx, frame in frames:
            res = self.detect(frame, confidence_threshold=confidence_threshold)
            dets = res.get("detections", [])
            per_frame_results[f_idx] = dets
            total_detections_across_video += len(dets)

            for d in dets:
                overall_summary[d["class"]] += 1

            if len(dets) > max_vehicles_in_single_frame:
                max_vehicles_in_single_frame = len(dets)
                best_vehicle_frame_idx = f_idx
                best_vehicle_detections = dets

        overall_summary["total"] = total_detections_across_video

        return {
            "success": True,
            "total_detections": total_detections_across_video,
            "max_per_frame": max_vehicles_in_single_frame,
            "best_frame_idx": best_vehicle_frame_idx,
            "best_frame_detections": best_vehicle_detections,
            "per_frame": per_frame_results,
            "summary": overall_summary,
        }

    def track_video_stream(
        self,
        video_input: str,
        confidence_threshold: float = 0.25,
        max_frames: int = 300,
    ) -> Dict[str, Any]:
        """
        Run real-time vehicle tracking + unique vehicle counting using ByteTrack.
        Processes video frames sequentially through YOLO11n and ByteTrack.
        Returns persistent tracking IDs, track histories, and exact unique vehicle counts.
        """
        cap = cv2.VideoCapture(video_input)
        if not cap.isOpened():
            return {
                "success": False,
                "error": f"Failed to open video file: {video_input}",
                "total_unique": 0,
                "unique_counts": {c: 0 for c in self.target_classes.values()},
                "track_history": {},
                "best_frame": None,
                "best_frame_idx": None,
                "best_frame_detections": [],
                "per_frame": {},
            }

        tracker = sv.ByteTrack(
            track_activation_threshold=confidence_threshold,
            lost_track_buffer=30,
            minimum_matching_threshold=0.8,
        )

        frame_idx = 0
        track_records: Dict[int, Dict[str, Any]] = {}
        per_frame_tracks: Dict[int, List[Dict[str, Any]]] = {}
        frames_cache: Dict[int, np.ndarray] = {}

        best_frame_idx: Optional[int] = None
        best_frame_count = 0
        best_frame_conf_sum = 0.0

        while cap.isOpened() and frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            h, w = frame.shape[:2]
            results = self.model(
                frame,
                classes=list(self.target_classes.keys()),
                conf=confidence_threshold,
                verbose=False,
            )[0]

            detections = sv.Detections.from_ultralytics(results)
            tracked = tracker.update_with_detections(detections)

            frame_tracked_items: List[Dict[str, Any]] = []
            frame_conf_sum = 0.0

            for xyxy, mask, conf, class_id, tracker_id, data in tracked:
                cls_id = int(class_id)
                cls_name = self.target_classes.get(cls_id, results.names.get(cls_id, "Vehicle").capitalize())
                tid = int(tracker_id)
                c_conf = float(conf)
                frame_conf_sum += c_conf

                x_min = max(0.0, min(float(w), float(xyxy[0])))
                y_min = max(0.0, min(float(h), float(xyxy[1])))
                x_max = max(0.0, min(float(w), float(xyxy[2])))
                y_max = max(0.0, min(float(h), float(xyxy[3])))
                box_w = max(0.0, x_max - x_min)
                box_h = max(0.0, y_max - y_min)

                if tid not in track_records:
                    track_records[tid] = {
                        "track_id": tid,
                        "class_counts": Counter(),
                        "max_conf": c_conf,
                        "frames": [],
                    }
                track_records[tid]["class_counts"][cls_name] += 1
                track_records[tid]["max_conf"] = max(track_records[tid]["max_conf"], c_conf)
                track_records[tid]["frames"].append(frame_idx)

                frame_tracked_items.append({
                    "track_id": tid,
                    "tracker_id": tid,
                    "class": cls_name,
                    "label": f"{cls_name} #{tid}",
                    "confidence": round(c_conf, 4),
                    "confidence_pct": int(round(c_conf * 100.0)),
                    "type": "vehicle",
                    "bbox": {
                        "x_min": round(x_min, 2),
                        "y_min": round(y_min, 2),
                        "x_max": round(x_max, 2),
                        "y_max": round(y_max, 2),
                        "x": round(x_min + box_w / 2.0, 2),
                        "y": round(y_min + box_h / 2.0, 2),
                        "width": round(box_w, 2),
                        "height": round(box_h, 2),
                        "box_2d": [round(x_min, 2), round(y_min, 2), round(x_max, 2), round(y_max, 2)],
                    },
                })

            per_frame_tracks[frame_idx] = frame_tracked_items

            # Select frame with highest number of active tracked vehicles (tiebreak by confidence)
            if len(frame_tracked_items) > best_frame_count or (
                len(frame_tracked_items) == best_frame_count and frame_conf_sum > best_frame_conf_sum
            ):
                best_frame_count = len(frame_tracked_items)
                best_frame_conf_sum = frame_conf_sum
                best_frame_idx = frame_idx
                frames_cache = {frame_idx: frame.copy()}

            frame_idx += 1

        cap.release()

        # Compute exact unique counts based on persistent track IDs (NOT raw detections)
        unique_counts = {c: 0 for c in self.target_classes.values()}
        unique_counts["total"] = 0

        track_persistence_examples = []
        for tid, rec in track_records.items():
            majority_class = rec["class_counts"].most_common(1)[0][0]
            rec["majority_class"] = majority_class
            if majority_class in unique_counts:
                unique_counts[majority_class] += 1
            else:
                unique_counts[majority_class] = 1
            unique_counts["total"] += 1

            if len(rec["frames"]) >= 2 and len(track_persistence_examples) < 8:
                track_persistence_examples.append({
                    "track_id": tid,
                    "class": majority_class,
                    "frame_count": len(rec["frames"]),
                    "frames": rec["frames"][:6],
                    "max_conf": round(rec["max_conf"], 2),
                })

        best_frame = frames_cache.get(best_frame_idx) if best_frame_idx is not None else None
        best_frame_detections = per_frame_tracks.get(best_frame_idx, []) if best_frame_idx is not None else []
        best_frame_detections.sort(key=lambda d: d.get("confidence", 0.0), reverse=True)

        return {
            "success": True,
            "total_frames_processed": frame_idx,
            "total_unique": unique_counts["total"],
            "unique_counts": unique_counts,
            "track_records": {
                tid: {
                    "track_id": tid,
                    "class": rec["majority_class"],
                    "frame_count": len(rec["frames"]),
                    "first_frame": rec["frames"][0],
                    "last_frame": rec["frames"][-1],
                    "max_conf": round(rec["max_conf"], 4),
                }
                for tid, rec in track_records.items()
            },
            "track_persistence_examples": track_persistence_examples,
            "best_frame": best_frame,
            "best_frame_idx": best_frame_idx,
            "best_frame_detections": best_frame_detections,
            "per_frame": per_frame_tracks,
        }

    CLASS_COLORS: Dict[str, Tuple[int, int, int]] = {
        "Car": (0, 180, 255),        # Vibrant Gold/Amber in BGR
        "Motorcycle": (255, 0, 255), # Magenta in BGR
        "Bus": (0, 230, 115),        # Bright Green in BGR
        "Truck": (255, 140, 0),      # Deep Sky Blue in BGR
    }

    def annotate_frame(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        default_color: Tuple[int, int, int] = (255, 200, 0),
    ) -> np.ndarray:
        """
        Draw bounding boxes and labels for vehicles on the frame with class-specific colors.
        Displays persistent tracking ID (e.g. 'Car #12 — 84%') when available.
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        for det in detections:
            cls_name = det.get("class", "Vehicle")
            color = self.CLASS_COLORS.get(cls_name, default_color)

            bbox = det.get("bbox", {})
            x1 = int(round(bbox.get("x_min", 0)))
            y1 = int(round(bbox.get("y_min", 0)))
            x2 = int(round(bbox.get("x_max", 0)))
            y2 = int(round(bbox.get("y_max", 0)))

            # Clamp boundaries
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w, x2))
            y2 = max(0, min(h, y2))

            track_id = det.get("track_id") or det.get("tracker_id")
            conf_pct = det.get("confidence_pct", int(det.get("confidence", 0.0) * 100))
            if track_id is not None:
                label = f"{cls_name} #{track_id} — {int(conf_pct)}%"
            else:
                label = f"{cls_name} {int(conf_pct)}%"

            # Bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Label tag
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.52
            thickness = 1
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

            tag_y1 = max(0, y1 - th - 8)
            tag_y2 = y1
            tag_x2 = min(w, x1 + tw + 8)

            cv2.rectangle(annotated, (x1, tag_y1), (tag_x2, tag_y2), color, -1)
            cv2.putText(
                annotated,
                label,
                (x1 + 4, tag_y2 - 4),
                font,
                font_scale,
                (0, 0, 0),  # Black text for maximum readability
                thickness,
                cv2.LINE_AA,
            )

        return annotated


# Global singleton instance
_vehicle_detector_instance: Optional[VehicleDetector] = None

def get_vehicle_detector() -> VehicleDetector:
    """
    Get or create the singleton VehicleDetector instance.
    """
    global _vehicle_detector_instance
    if _vehicle_detector_instance is None:
        _vehicle_detector_instance = VehicleDetector()
    return _vehicle_detector_instance

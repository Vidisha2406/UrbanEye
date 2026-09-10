import os
import cv2
import numpy as np
from typing import Union, List, Dict, Any, Tuple
from backend.inference.vehicle_detector import get_vehicle_detector, VehicleDetector

class TrafficAnalyzer:
    """
    Traffic & Mobility Intelligence Analyzer for UrbanEye.
    Reuses the existing verified YOLO11n vehicle detector to perform
    single-image vehicle classification, counting, density calculation,
    and deterministic congestion evaluation.
    """

    def __init__(self, vehicle_detector: VehicleDetector = None):
        self.vehicle_detector = vehicle_detector or get_vehicle_detector()
        self.model_name = "YOLO11n Traffic Analyzer"

    def calculate_traffic_density(self, vehicle_count: int, occupancy_ratio: float = 0.0) -> str:
        """
        Explicit, documented prototype rule for traffic density:
        - LOW: < 4 vehicles (or occupancy < 10% if count is 0)
        - MODERATE: 4 to 7 vehicles
        - HIGH: 8 to 14 vehicles
        - SEVERE: >= 15 vehicles
        """
        if vehicle_count >= 15:
            return "SEVERE"
        elif vehicle_count >= 8:
            return "HIGH"
        elif vehicle_count >= 4:
            return "MODERATE"
        else:
            return "LOW"

    def calculate_congestion_level(self, traffic_density: str) -> str:
        """
        Deterministic mapping from traffic density to congestion state:
        - LOW -> FREE FLOW
        - MODERATE -> MODERATE
        - HIGH -> HIGH
        - SEVERE -> SEVERE
        """
        mapping = {
            "LOW": "FREE FLOW",
            "MODERATE": "MODERATE",
            "HIGH": "HIGH",
            "SEVERE": "SEVERE",
        }
        return mapping.get(traffic_density.upper(), "FREE FLOW")

    def analyze(
        self,
        image_input: Union[str, np.ndarray],
        confidence_threshold: float = 0.25
    ) -> Dict[str, Any]:
        """
        Analyze a real traffic photograph:
        1. Run YOLO11n detection (Car, Motorcycle, Bus, Truck).
        2. Compute vehicle counts and per-class mix.
        3. Evaluate image occupancy ratio.
        4. Calculate density & congestion deterministically.
        """
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image not found at: {image_input}")
            frame = cv2.imread(image_input)
            if frame is None:
                raise ValueError(f"Could not read image at: {image_input}")
        elif isinstance(image_input, np.ndarray):
            frame = image_input
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        h, w = frame.shape[:2]
        total_image_area = float(h * w)

        # 1. Run YOLO11n detection
        v_res = self.vehicle_detector.detect(frame, confidence_threshold=confidence_threshold)
        detections = v_res.get("detections", [])
        total_vehicles = len(detections)

        # 2. Count per vehicle class
        counts_by_class: Dict[str, int] = {
            "Car": 0,
            "Motorcycle": 0,
            "Bus": 0,
            "Truck": 0,
        }
        total_bbox_area = 0.0
        conf_sum = 0.0

        for d in detections:
            cls = d.get("class", "Car")
            if cls in counts_by_class:
                counts_by_class[cls] += 1
            else:
                counts_by_class[cls] = 1

            bbox = d.get("bbox", {})
            bw = float(bbox.get("width", 0.0))
            bh = float(bbox.get("height", 0.0))
            total_bbox_area += (bw * bh)
            conf_sum += float(d.get("confidence", 0.5))

        # 3. Compute vehicle mix percentages
        vehicle_mix: Dict[str, str] = {}
        for cls, cnt in counts_by_class.items():
            if total_vehicles > 0:
                pct = round((cnt / total_vehicles) * 100.0)
                vehicle_mix[cls] = f"{pct}%"
            else:
                vehicle_mix[cls] = "0%"

        # 4. Image occupancy ratio
        occupancy_ratio = min(1.0, total_bbox_area / total_image_area) if total_image_area > 0 else 0.0

        # 5. Density and Congestion calculation
        density = self.calculate_traffic_density(total_vehicles, occupancy_ratio)
        congestion = self.calculate_congestion_level(density)
        avg_conf = (conf_sum / total_vehicles) if total_vehicles > 0 else 0.90

        return {
            "success": True,
            "model": self.model_name,
            "vehicle_count": total_vehicles,
            "vehicle_counts_by_class": counts_by_class,
            "vehicle_mix": vehicle_mix,
            "traffic_density": density,
            "congestion_level": congestion,
            "occupancy_ratio": round(occupancy_ratio, 4),
            "derived_confidence": round(avg_conf, 4),
            "detections": detections,
            "frame_dimensions": {"width": w, "height": h},
        }

    def annotate_frame(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        telemetry: Dict[str, Any]
    ) -> np.ndarray:
        """
        Draws clear bounding boxes with class and model confidence, plus a clean
        top banner showing real traffic telemetry (count, density, congestion).
        Does NOT draw tracking IDs or fake unique IDs.
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        # Class Color Mapping (BGR)
        class_colors = {
            "Car": (240, 140, 0),        # Vibrant Blue/Cyan
            "Bus": (220, 0, 200),        # Vibrant Purple/Magenta
            "Truck": (0, 165, 255),      # Amber/Orange
            "Motorcycle": (80, 210, 0),  # Emerald Green
        }

        # 1. Draw vehicle bounding boxes
        for det in detections:
            cls = det.get("class", "Car")
            color = class_colors.get(cls, (240, 140, 0))
            bbox = det.get("bbox", {})
            x_min = int(round(bbox.get("x_min", 0)))
            y_min = int(round(bbox.get("y_min", 0)))
            x_max = int(round(bbox.get("x_max", 0)))
            y_max = int(round(bbox.get("y_max", 0)))

            x_min = max(0, min(w - 1, x_min))
            y_min = max(0, min(h - 1, y_min))
            x_max = max(0, min(w, x_max))
            y_max = max(0, min(h, y_max))

            conf_pct = int(round(det.get("confidence", 0.8) * 100))
            label = f"{cls} — {conf_pct}%"

            # Bounding box
            cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), color, 2)

            # High-tech corner brackets
            c_len = min(18, max(8, (x_max - x_min) // 5))
            c_thick = 2
            cv2.line(annotated, (x_min, y_min), (x_min + c_len, y_min), (255, 255, 255), c_thick)
            cv2.line(annotated, (x_min, y_min), (x_min, y_min + c_len), (255, 255, 255), c_thick)
            cv2.line(annotated, (x_max, y_min), (x_max - c_len, y_min), (255, 255, 255), c_thick)
            cv2.line(annotated, (x_max, y_min), (x_max, y_min + c_len), (255, 255, 255), c_thick)
            cv2.line(annotated, (x_min, y_max), (x_min + c_len, y_max), (255, 255, 255), c_thick)
            cv2.line(annotated, (x_min, y_max), (x_min, y_max - c_len), (255, 255, 255), c_thick)
            cv2.line(annotated, (x_max, y_max), (x_max - c_len, y_max), (255, 255, 255), c_thick)
            cv2.line(annotated, (x_max, y_max), (x_max, y_max - c_len), (255, 255, 255), c_thick)

            # Label banner
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

            tag_y2 = y_min if y_min >= th + 8 else min(h, y_min + th + 10)
            tag_y1 = max(0, tag_y2 - th - 6)
            tag_x2 = min(w, x_min + tw + 10)

            cv2.rectangle(annotated, (x_min, tag_y1), (tag_x2, tag_y2), color, -1)
            cv2.putText(
                annotated,
                label,
                (x_min + 5, tag_y2 - 4),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

        # 2. Top Telemetry Overlay Bar
        bar_height = 42
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_height), (20, 28, 32), -1)
        cv2.addWeighted(overlay, 0.85, annotated, 0.15, 0, annotated)

        # Text on telemetry bar
        v_count = telemetry.get("vehicle_count", len(detections))
        density = telemetry.get("traffic_density", "HIGH")
        congestion = telemetry.get("congestion_level", "HIGH")
        telemetry_text = (
            f"URBANEYE TRAFFIC INTELLIGENCE | Vehicles: {v_count} | "
            f"Density: {density} | Congestion: {congestion}"
        )

        cv2.putText(
            annotated,
            telemetry_text,
            (14, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 240, 255),
            2,
            cv2.LINE_AA,
        )

        return annotated


# Singleton helper
_traffic_analyzer_instance = None

def get_traffic_analyzer() -> TrafficAnalyzer:
    global _traffic_analyzer_instance
    if _traffic_analyzer_instance is None:
        _traffic_analyzer_instance = TrafficAnalyzer()
    return _traffic_analyzer_instance

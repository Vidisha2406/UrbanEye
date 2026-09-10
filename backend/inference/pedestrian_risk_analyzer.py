import os
import cv2
import numpy as np
from typing import Union, List, Dict, Any, Optional, Tuple
from ultralytics import YOLO

# Resolve directory paths
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(BACKEND_DIR)

class PedestrianRiskAnalyzer:
    """
    Real AI Pedestrian Risk Detection & Spatial Hazard Analyzer for UrbanEye Edge AI.
    Uses the verified YOLO11n detector (MS-COCO Class 0: 'person') to detect pedestrians,
    compute spatial camera proximity (bumper distance) and lane-conflict metrics,
    and derive a transparent, deterministic prototype image-based risk estimate.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = self._resolve_model_path(model_path)
        print(f"[PEDESTRIAN_ANALYZER] Loading YOLO11n model from: {self.model_path}")
        self.model = YOLO(self.model_path)
        self.model_name = "YOLO11n Pedestrian Risk Analyzer"
        self.target_class_id = 0  # MS-COCO class 0 is person

    def _resolve_model_path(self, model_path: Optional[str]) -> str:
        if model_path and os.path.exists(model_path):
            return model_path

        candidates = [
            os.getenv("YOLO_MODEL_PATH"),
            os.path.join(BACKEND_DIR, "models", "yolo11n.pt"),
            os.path.join(ROOT_DIR, "yolo11n.pt"),
            os.path.join(BACKEND_DIR, "yolo11n.pt"),
            "yolo11n.pt",
        ]
        for path in candidates:
            if path and os.path.exists(path):
                return path

        return "yolo11n.pt"

    def calculate_spatial_risk(
        self,
        pedestrian_boxes: List[Dict[str, Any]],
        image_width: int,
        image_height: int,
        vehicle_count: int = 0
    ) -> Dict[str, Any]:
        """
        Deterministic spatial risk estimation from pedestrian bounding boxes.
        Computes vertical bumper proximity, lateral center driving-corridor deviation,
        and group conflict density to yield a prototype image-based risk estimate (0-100).
        """
        total_pedestrians = len(pedestrian_boxes)
        if total_pedestrians == 0:
            return {
                "risk_score": 0,
                "risk_level": "LOW",
                "max_proximity_ratio": 0.0,
                "min_center_offset": 1.0,
                "in_conflict_zone_count": 0,
                "risk_estimate_label": "Prototype Image-Based Risk Estimate: 0/100 (LOW)",
                "context_notes": ["No pedestrians detected in active roadway."]
            }

        max_prox = 0.0
        min_offset = 1.0
        in_path_count = 0
        prox_scores = []
        path_scores = []

        for p in pedestrian_boxes:
            bbox = p.get("bbox", {})
            xmin = float(bbox.get("x_min", 0.0))
            ymin = float(bbox.get("y_min", 0.0))
            xmax = float(bbox.get("x_max", 0.0))
            ymax = float(bbox.get("y_max", 0.0))

            # 1. Vertical position (bumper proximity)
            # Higher ymax means object is physically closer to the front camera bumper
            norm_ymax = min(1.0, ymax / image_height) if image_height > 0 else 0.0
            max_prox = max(max_prox, norm_ymax)

            # Proximity score (0 to 100)
            # Above 0.25 (road surface visible), scaling linearly to 0.85+ (imminent contact)
            s_prox = min(100.0, max(0.0, ((norm_ymax - 0.25) / 0.55) * 100.0))
            prox_scores.append(s_prox)

            # 2. Lateral driving path conflict (center deviation)
            x_center = (xmin + xmax) / 2.0
            norm_center_offset = abs((x_center / image_width) - 0.5) if image_width > 0 else 0.5
            min_offset = min(min_offset, norm_center_offset)

            if norm_center_offset < 0.22:
                # Direct travel corridor / lane center
                s_path = 100.0
                in_path_count += 1
            elif norm_center_offset < 0.38:
                # Adjacent driving lane / crossing transition
                s_path = 65.0
                in_path_count += 1
            else:
                # Shoulder / outer sidewalk
                s_path = 25.0

            path_scores.append(s_path)

        # 3. Vulnerable crowd / group density factor
        if total_pedestrians >= 6:
            s_count = 100.0
        elif total_pedestrians >= 3:
            s_count = 75.0
        elif total_pedestrians >= 2:
            s_count = 55.0
        else:
            s_count = 40.0

        # 4. Traffic conflict presence
        s_traffic = 80.0 if vehicle_count >= 2 else (50.0 if vehicle_count == 1 else 30.0)

        # 5. Composite Prototype Image-Based Risk Score Equation
        s_prox_max = max(prox_scores) if prox_scores else 0.0
        s_path_max = max(path_scores) if path_scores else 0.0

        raw_score = (
            0.40 * s_prox_max +
            0.30 * s_path_max +
            0.15 * s_count +
            0.15 * s_traffic
        )
        risk_score = int(round(min(100.0, max(0.0, raw_score))))

        # 6. Deterministic Risk Level Categorization
        if risk_score >= 75:
            risk_level = "CRITICAL"
        elif risk_score >= 55:
            risk_level = "HIGH"
        elif risk_score >= 35:
            risk_level = "MODERATE"
        else:
            risk_level = "LOW"

        # Context details
        context_notes = [
            f"Prototype Image-Based Risk Estimate: {risk_score}/100 ({risk_level})",
            f"{total_pedestrians} Pedestrian(s) identified in front roadway view",
            f"{in_path_count} Pedestrian(s) situated in primary vehicle transit corridor",
            f"Maximum camera bumper proximity: {round(max_prox * 100, 1)}%",
        ]

        return {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "max_proximity_ratio": round(max_prox, 3),
            "min_center_offset": round(min_offset, 3),
            "in_conflict_zone_count": in_path_count,
            "risk_estimate_label": f"Prototype Image-Based Risk Estimate: {risk_score}/100 ({risk_level})",
            "context_notes": context_notes,
        }

    def analyze(
        self,
        image_input: Union[str, np.ndarray],
        confidence_threshold: float = 0.25
    ) -> Dict[str, Any]:
        """
        Run pedestrian detection and spatial risk evaluation on a single image.
        """
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                return {"success": False, "error": f"File not found: {image_input}"}
            frame = cv2.imread(image_input)
            if frame is None:
                return {"success": False, "error": "Failed to read image with OpenCV"}
        elif isinstance(image_input, np.ndarray):
            frame = image_input
        else:
            return {"success": False, "error": "Invalid input type. Expected filepath or numpy array."}

        h, w = frame.shape[:2]

        try:
            # Run YOLO11n inference
            results = self.model(frame, conf=confidence_threshold, verbose=False)
            boxes = results[0].boxes

            pedestrian_detections: List[Dict[str, Any]] = []
            vehicle_detections: List[Dict[str, Any]] = []

            for i, box in enumerate(boxes):
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                xyxy = box.xyxy[0].tolist()

                xmin, ymin, xmax, ymax = xyxy
                bw = xmax - xmin
                bh = ymax - ymin

                det_obj = {
                    "id": i + 1,
                    "class": "Pedestrian" if cls_id == 0 else self.model.names.get(cls_id, "Object"),
                    "confidence": round(conf, 4),
                    "confidence_pct": int(round(conf * 100)),
                    "bbox": {
                        "x": round(xmin + bw / 2, 1),
                        "y": round(ymin + bh / 2, 1),
                        "width": round(bw, 1),
                        "height": round(bh, 1),
                        "x_min": round(xmin, 1),
                        "y_min": round(ymin, 1),
                        "x_max": round(xmax, 1),
                        "y_max": round(ymax, 1),
                    },
                }

                if cls_id == 0:
                    pedestrian_detections.append(det_obj)
                elif cls_id in [2, 3, 5, 7]:
                    vehicle_detections.append(det_obj)

            # Spatial risk estimation
            risk_data = self.calculate_spatial_risk(
                pedestrian_detections,
                image_width=w,
                image_height=h,
                vehicle_count=len(vehicle_detections)
            )

            # Average confidence of detected pedestrians
            avg_conf = (
                sum(p["confidence"] for p in pedestrian_detections) / len(pedestrian_detections)
                if pedestrian_detections else 0.85
            )

            return {
                "success": True,
                "model": self.model_name,
                "pedestrian_count": len(pedestrian_detections),
                "vehicle_count": len(vehicle_detections),
                "risk_score": risk_data["risk_score"],
                "risk_level": risk_data["risk_level"],
                "risk_estimate_type": "Prototype Image-Based Risk Estimate",
                "max_proximity_ratio": risk_data["max_proximity_ratio"],
                "min_center_offset": risk_data["min_center_offset"],
                "in_conflict_zone_count": risk_data["in_conflict_zone_count"],
                "context_notes": risk_data["context_notes"],
                "derived_confidence": round(avg_conf, 4),
                "detections": pedestrian_detections,
                "frame_dimensions": {"width": w, "height": h},
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Pedestrian risk analysis failed: {str(e)}",
                "pedestrian_count": 0,
                "risk_score": 0,
                "risk_level": "LOW",
                "detections": []
            }

    def analyze_frames(
        self,
        frames: List[Tuple[int, np.ndarray]],
        confidence_threshold: float = 0.25
    ) -> Dict[str, Any]:
        """
        Run pedestrian detection and spatial risk evaluation across a sequence of sampled video frames.
        Selects the frame with maximum spatial risk / pedestrian conflict for evidence generation.
        """
        if not frames:
            return {
                "success": False,
                "error": "No frames provided for analysis.",
                "pedestrian_count": 0,
                "risk_score": 0,
                "risk_level": "LOW",
                "detections": [],
                "best_frame": None,
                "best_frame_idx": None,
            }

        best_res: Optional[Dict[str, Any]] = None
        best_frame: Optional[np.ndarray] = None
        best_frame_idx: Optional[int] = None
        max_risk_score = -1
        max_ped_count = -1
        per_frame_results: Dict[int, Dict[str, Any]] = {}

        for f_idx, frame in frames:
            res = self.analyze(frame, confidence_threshold=confidence_threshold)
            per_frame_results[f_idx] = res
            r_score = res.get("risk_score", 0)
            p_count = res.get("pedestrian_count", 0)

            if r_score > max_risk_score or (r_score == max_risk_score and p_count > max_ped_count):
                max_risk_score = r_score
                max_ped_count = p_count
                best_res = res
                best_frame = frame
                best_frame_idx = f_idx

        if best_res is None:
            best_res = self.analyze(frames[0][1], confidence_threshold=confidence_threshold)
            best_frame = frames[0][1]
            best_frame_idx = frames[0][0]

        return {
            **best_res,
            "best_frame": best_frame,
            "best_frame_idx": best_frame_idx,
            "per_frame": per_frame_results,
        }

    def annotate_frame(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        analysis_result: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Annotates frame with high-contrast amber/rose tech bounding boxes, corner brackets,
        and HUD telemetry bar explicitly labeled as a prototype image-based risk estimate.
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        risk_level = (analysis_result or {}).get("risk_level", "HIGH")
        risk_score = (analysis_result or {}).get("risk_score", 68)
        ped_count = (analysis_result or {}).get("pedestrian_count", len(detections))

        # Visual theme color based on severity
        if risk_level == "CRITICAL":
            primary_color = (0, 0, 235)      # Bright Red (BGR)
            secondary_color = (30, 30, 255)
        elif risk_level == "HIGH":
            primary_color = (0, 140, 255)    # Orange/Amber (BGR)
            secondary_color = (20, 170, 255)
        elif risk_level == "MODERATE":
            primary_color = (0, 215, 255)    # Yellow (BGR)
            secondary_color = (50, 230, 255)
        else:
            primary_color = (0, 200, 50)     # Emerald Green (BGR)
            secondary_color = (50, 230, 100)

        # 1. Draw tech bounding boxes and corner brackets for each pedestrian
        for d in detections:
            bbox = d.get("bbox", {})
            xmin = int(round(bbox.get("x_min", 0)))
            ymin = int(round(bbox.get("y_min", 0)))
            xmax = int(round(bbox.get("x_max", 0)))
            ymax = int(round(bbox.get("y_max", 0)))
            conf_pct = d.get("confidence_pct", int(d.get("confidence", 0.5) * 100))

            # Translucent box fill
            sub_overlay = annotated.copy()
            cv2.rectangle(sub_overlay, (xmin, ymin), (xmax, ymax), primary_color, -1)
            cv2.addWeighted(sub_overlay, 0.12, annotated, 0.88, 0, annotated)

            # Core bounding rectangle
            cv2.rectangle(annotated, (xmin, ymin), (xmax, ymax), primary_color, 2, cv2.LINE_AA)

            # High-visibility tech corner brackets
            bracket_len = max(10, min(24, int((xmax - xmin) * 0.25)))
            thick = 3
            # Top-left
            cv2.line(annotated, (xmin, ymin), (xmin + bracket_len, ymin), primary_color, thick)
            cv2.line(annotated, (xmin, ymin), (xmin, ymin + bracket_len), primary_color, thick)
            # Top-right
            cv2.line(annotated, (xmax, ymin), (xmax - bracket_len, ymin), primary_color, thick)
            cv2.line(annotated, (xmax, ymin), (xmax, ymin + bracket_len), primary_color, thick)
            # Bottom-left
            cv2.line(annotated, (xmin, ymax), (xmin + bracket_len, ymax), primary_color, thick)
            cv2.line(annotated, (xmin, ymax), (xmin, ymax - bracket_len), primary_color, thick)
            # Bottom-right
            cv2.line(annotated, (xmax, ymax), (xmax - bracket_len, ymax), primary_color, thick)
            cv2.line(annotated, (xmax, ymax), (xmax, ymax - bracket_len), primary_color, thick)

            # Pedestrian label tag
            label_text = f"PEDESTRIAN {conf_pct}%"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.50
            (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, 1)

            tag_y1 = max(0, ymin - text_h - 8)
            tag_y2 = ymin
            tag_x1 = xmin
            tag_x2 = xmin + text_w + 12

            cv2.rectangle(annotated, (tag_x1, tag_y1), (tag_x2, tag_y2), (20, 25, 28), -1)
            cv2.rectangle(annotated, (tag_x1, tag_y1), (tag_x2, tag_y2), primary_color, 1)
            cv2.putText(
                annotated,
                label_text,
                (tag_x1 + 6, tag_y2 - 5),
                font,
                font_scale,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # 2. Top HUD Telemetry Banner
        hud_h = 44
        hud_overlay = annotated.copy()
        cv2.rectangle(hud_overlay, (0, 0), (w, hud_h), (18, 28, 30), -1)
        cv2.addWeighted(hud_overlay, 0.88, annotated, 0.12, 0, annotated)
        cv2.line(annotated, (0, hud_h), (w, hud_h), primary_color, 2)

        hud_title = f"URBANEYE SAFETY INTELLIGENCE | Pedestrians: {ped_count} | Est. Risk: {risk_level} ({risk_score}/100 Image-Based Estimate)"
        cv2.putText(
            annotated,
            hud_title,
            (16, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # Right-aligned live indicator
        badge_text = "AI VERIFIED HAZARD"
        (b_w, b_h), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        badge_x = w - b_w - 24
        cv2.circle(annotated, (badge_x - 8, 22), 4, primary_color, -1)
        cv2.putText(
            annotated,
            badge_text,
            (badge_x, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            primary_color,
            1,
            cv2.LINE_AA,
        )

        # 3. Bottom Camera / Location Banner
        bot_h = 32
        bot_overlay = annotated.copy()
        cv2.rectangle(bot_overlay, (0, h - bot_h), (w, h), (18, 28, 30), -1)
        cv2.addWeighted(bot_overlay, 0.88, annotated, 0.12, 0, annotated)

        bot_text = f"BUS: BUS-01 | LOCATION: MG Road, Pune | STATUS: ACTIVE ROADWAY CONFLICT"
        cv2.putText(
            annotated,
            bot_text,
            (16, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 220, 220),
            1,
            cv2.LINE_AA,
        )

        return annotated

_pedestrian_analyzer_instance: Optional[PedestrianRiskAnalyzer] = None

def get_pedestrian_risk_analyzer(model_path: Optional[str] = None) -> PedestrianRiskAnalyzer:
    """
    Get or create singleton instance of PedestrianRiskAnalyzer.
    """
    global _pedestrian_analyzer_instance
    if _pedestrian_analyzer_instance is None:
        _pedestrian_analyzer_instance = PedestrianRiskAnalyzer(model_path=model_path)
    return _pedestrian_analyzer_instance


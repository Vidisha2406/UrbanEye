import os
import cv2
import numpy as np
from typing import Union, List, Dict, Any
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient, InferenceConfiguration

# Load environment variables
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(backend_dir, ".env")
load_dotenv(dotenv_path=env_path)

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
ROBOFLOW_INFRASTRUCTURE_MODEL_ID = os.getenv(
    "ROBOFLOW_INFRASTRUCTURE_MODEL_ID",
    "faded-or-damaged-signs-detection-u7arv/1"
)

class InfrastructureDetector:
    """
    Roboflow-backed Infrastructure Deficiency Detection Client for UrbanEye Edge AI.
    Connects to the Roboflow Inference API using hosted model 'faded-or-damaged-signs-detection-u7arv/1'.
    Detects damaged road and traffic signs, returning bounding boxes, confidence, and severity.
    """

    def __init__(self, api_key: str = None, model_id: str = None):
        self.api_key = api_key or ROBOFLOW_API_KEY
        self.model_id = model_id or ROBOFLOW_INFRASTRUCTURE_MODEL_ID

        if not self.api_key:
            raise ValueError("ROBOFLOW_API_KEY is not set. Please configure it in backend/.env")

        # Initialize Roboflow Inference Client with standard secure configuration
        self.client = InferenceHTTPClient(
            api_url="https://detect.roboflow.com",
            api_key=self.api_key
        ).configure(InferenceConfiguration(api_key_transport="header"))

    def detect(self, image_input: Union[str, np.ndarray], confidence_threshold: float = 0.25) -> Dict[str, Any]:
        """
        Run infrastructure deficiency detection on a single image (filepath or numpy BGR array).

        Args:
            image_input: File path to image, or numpy ndarray (cv2 image).
            confidence_threshold: Minimum confidence score to retain detection (0.0 to 1.0).

        Returns:
            Dictionary with parsed detections, bounding boxes, and summary metadata.
        """
        temp_file_created = False
        target_path = None

        try:
            if isinstance(image_input, np.ndarray):
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
                    target_path = tf.name
                    temp_file_created = True
                cv2.imwrite(target_path, image_input)
            elif isinstance(image_input, str):
                target_path = image_input
            else:
                raise ValueError(f"Unsupported image input type: {type(image_input)}")

            # Call Roboflow hosted model
            response = self.client.infer(target_path, model_id=self.model_id)

            predictions = []
            if isinstance(response, dict):
                predictions = response.get("predictions", [])
            elif isinstance(response, list) and len(response) > 0 and isinstance(response[0], dict):
                predictions = response[0].get("predictions", [])

            detections = []
            for pred in predictions:
                conf = float(pred.get("confidence", 0.0))
                if conf < confidence_threshold:
                    continue

                raw_class = pred.get("class", "Damaged")
                # Format class display: 'Damaged' -> 'Damaged Sign'
                if raw_class.lower() == "damaged":
                    formatted_class = "Damaged Sign"
                elif "sign" in raw_class.lower():
                    formatted_class = raw_class.strip()
                else:
                    formatted_class = f"{raw_class.capitalize()} Sign"

                cx = float(pred.get("x", 0.0))
                cy = float(pred.get("y", 0.0))
                w = float(pred.get("width", 0.0))
                h = float(pred.get("height", 0.0))

                x_min = max(0.0, cx - w / 2.0)
                y_min = max(0.0, cy - h / 2.0)
                x_max = cx + w / 2.0
                y_max = cy + h / 2.0

                detections.append({
                    "class": formatted_class,
                    "raw_class": raw_class,
                    "confidence": conf,
                    "confidence_pct": round(conf * 100.0, 1),
                    "type": "infrastructure",
                    "category": "Infrastructure Deficiencies",
                    "bbox": {
                        "x": cx,
                        "y": cy,
                        "width": w,
                        "height": h,
                        "x_min": x_min,
                        "y_min": y_min,
                        "x_max": x_max,
                        "y_max": y_max,
                    }
                })

            # Sort by confidence descending
            detections.sort(key=lambda d: d["confidence"], reverse=True)

            return {
                "success": True,
                "model_id": self.model_id,
                "count": len(detections),
                "detections": detections,
                "raw_predictions": predictions
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "count": 0,
                "detections": []
            }
        finally:
            if temp_file_created and target_path and os.path.exists(target_path):
                try:
                    os.remove(target_path)
                except OSError:
                    pass

    def annotate_frame(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """
        Draws vibrant amber bounding boxes, high-tech corner brackets, and labels on frame.
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        for det in detections:
            bbox = det.get("bbox", {})
            x_min = int(round(bbox.get("x_min", 0)))
            y_min = int(round(bbox.get("y_min", 0)))
            x_max = int(round(bbox.get("x_max", 0)))
            y_max = int(round(bbox.get("y_max", 0)))

            # Clamp to frame dimensions
            x_min = max(0, min(w - 1, x_min))
            y_min = max(0, min(h - 1, y_min))
            x_max = max(0, min(w, x_max))
            y_max = max(0, min(h, y_max))

            conf_pct = int(round(det.get("confidence_pct", det.get("confidence", 0.9) * 100)))
            class_name = det.get("class", "Damaged Sign")
            label = f"{class_name} {conf_pct}%"

            # Primary Color: Amber-Orange (BGR: (0, 140, 255))
            box_color = (0, 140, 255)
            accent_color = (0, 240, 255)

            # 1. Main Bounding Box
            cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), box_color, 3)

            # 2. Tech Corner Accents
            c_len = min(22, max(10, (x_max - x_min) // 5))
            c_thick = 3
            # Top-Left
            cv2.line(annotated, (x_min, y_min), (x_min + c_len, y_min), accent_color, c_thick)
            cv2.line(annotated, (x_min, y_min), (x_min, y_min + c_len), accent_color, c_thick)
            # Top-Right
            cv2.line(annotated, (x_max, y_min), (x_max - c_len, y_min), accent_color, c_thick)
            cv2.line(annotated, (x_max, y_min), (x_max, y_min + c_len), accent_color, c_thick)
            # Bottom-Left
            cv2.line(annotated, (x_min, y_max), (x_min + c_len, y_max), accent_color, c_thick)
            cv2.line(annotated, (x_min, y_max), (x_min, y_max - c_len), accent_color, c_thick)
            # Bottom-Right
            cv2.line(annotated, (x_max, y_max), (x_max - c_len, y_max), accent_color, c_thick)
            cv2.line(annotated, (x_max, y_max), (x_max, y_max - c_len), accent_color, c_thick)

            # 3. Label text
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.55
            thickness = 2
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

            tag_y2 = y_min if y_min >= text_h + 10 else min(h, y_min + text_h + 12)
            tag_y1 = max(0, tag_y2 - text_h - 8)
            tag_x2 = min(w, x_min + text_w + 14)

            # Background rectangle for text
            cv2.rectangle(annotated, (x_min, tag_y1), (tag_x2, tag_y2), box_color, -1)
            # Label text
            cv2.putText(
                annotated,
                label,
                (x_min + 6, tag_y2 - 5),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

        return annotated

    def determine_severity(self, confidence: float, count: int = 1) -> str:
        """
        Dynamically determine severity based on model confidence and detection count.
        """
        if confidence >= 0.85 or count >= 3:
            return "CRITICAL"
        elif confidence >= 0.70 or count >= 2:
            return "HIGH"
        elif confidence >= 0.40:
            return "MEDIUM"
        else:
            return "LOW"


# Singleton instance helper
_infra_detector_instance = None

def get_infrastructure_detector() -> InfrastructureDetector:
    global _infra_detector_instance
    if _infra_detector_instance is None:
        _infra_detector_instance = InfrastructureDetector()
    return _infra_detector_instance

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
ROBOFLOW_MODEL_ID = os.getenv("ROBOFLOW_MODEL_ID", "potholes-1bo4b/1")

class PotholeDetector:
    """
    Roboflow-backed Pothole Detection Client for UrbanEye Edge AI.
    Connects to the Roboflow Inference API using model 'potholes-1bo4b/1'.
    """

    def __init__(self, api_key: str = None, model_id: str = None):
        self.api_key = api_key or ROBOFLOW_API_KEY
        self.model_id = model_id or ROBOFLOW_MODEL_ID

        if not self.api_key:
            raise ValueError("ROBOFLOW_API_KEY is not set. Please configure it in backend/.env")

        # Initialize Roboflow Inference Client
        self.client = InferenceHTTPClient(
            api_url="https://detect.roboflow.com",
            api_key=self.api_key
        ).configure(InferenceConfiguration(api_key_transport="legacy"))

    def detect(self, image_input: Union[str, np.ndarray], confidence_threshold: float = 0.25) -> Dict[str, Any]:
        """
        Run pothole detection on a single image (filepath or numpy BGR array).

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
                # Save numpy array temporarily for robust multi-platform inference_sdk compatibility
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
                    target_path = tf.name
                    temp_file_created = True
                cv2.imwrite(target_path, image_input)
            elif isinstance(image_input, str):
                img = cv2.imread(image_input)
                if img is not None:
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
                        target_path = tf.name
                        temp_file_created = True
                    cv2.imwrite(target_path, img)
                else:
                    target_path = image_input
            else:
                raise ValueError(f"Unsupported image input type: {type(image_input)}")

            # Call Roboflow model
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

                class_name = pred.get("class", "Pothole")
                # Format class cleanly (e.g. 'Pothole - v1 raw' -> 'Pothole')
                if "pothole" in class_name.lower():
                    formatted_class = "Pothole"
                else:
                    formatted_class = class_name.capitalize() if class_name else "Pothole"

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
                    "confidence": conf,
                    "confidence_pct": round(conf * 100.0, 1),
                    "type": "damage",
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

# Singleton instance helper
_detector_instance = None

def get_detector() -> PotholeDetector:
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = PotholeDetector()
    return _detector_instance

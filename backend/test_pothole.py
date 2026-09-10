import os
import sys
import numpy as np
import cv2
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient

# Load .env from backend directory
backend_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(backend_dir, ".env")
load_dotenv(dotenv_path=env_path)

api_key = os.getenv("ROBOFLOW_API_KEY")
model_id = os.getenv("ROBOFLOW_MODEL_ID", "potholes-1bo4b/1")

if not api_key:
    print("[ERROR] ROBOFLOW_API_KEY is not set in backend/.env")
    sys.exit(1)

print("[INFO] Initializing InferenceHTTPClient...")
client = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=api_key
)

# Create a test synthetic road frame
print("[INFO] Creating a test road image...")
test_img = np.zeros((480, 640, 3), dtype=np.uint8)
# Road gray background
test_img[:] = (60, 60, 60)
# Draw dark oval representing pothole
cv2.ellipse(test_img, (320, 300), (80, 40), 0, 0, 360, (20, 20, 20), -1)
# Crack lines
cv2.line(test_img, (250, 300), (220, 290), (10, 10, 10), 2)
cv2.line(test_img, (390, 300), (420, 310), (10, 10, 10), 2)

test_img_path = os.path.join(backend_dir, "test_sample.jpg")
cv2.imwrite(test_img_path, test_img)

try:
    print(f"[INFO] Sending test inference request to model: {model_id}...")
    result = client.infer(test_img_path, model_id=model_id)
    print("[SUCCESS] Roboflow API response received successfully!")
    print(f"[INFO] Response keys: {list(result.keys()) if isinstance(result, dict) else type(result)}")
    if isinstance(result, dict) and "predictions" in result:
        print(f"[INFO] Found {len(result['predictions'])} prediction(s)")
        for i, pred in enumerate(result["predictions"]):
            print(f"  - Prediction {i+1}: class={pred.get('class')}, conf={pred.get('confidence')}")
    else:
        print(f"[INFO] Full response: {result}")
finally:
    if os.path.exists(test_img_path):
        os.remove(test_img_path)

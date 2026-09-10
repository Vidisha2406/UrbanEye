import os
import sys
import cv2
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient, InferenceConfiguration

# Load .env from backend directory
backend_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(backend_dir, ".env")
load_dotenv(dotenv_path=env_path)

api_key = os.getenv("ROBOFLOW_API_KEY")
model_id = os.getenv("ROBOFLOW_INFRASTRUCTURE_MODEL_ID", "faded-or-damaged-signs-detection-u7arv/1")

if not api_key:
    print("[ERROR] ROBOFLOW_API_KEY is not set in backend/.env")
    sys.exit(1)

print("[INFO] Initializing Roboflow InferenceHTTPClient for infrastructure model...")
client = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=api_key
).configure(InferenceConfiguration(api_key_transport="header"))

repo_root = os.path.dirname(backend_dir)
sample_path = os.path.join(repo_root, "samples", "damaged_traffic_sign_sample.jpg")

if not os.path.exists(sample_path):
    print(f"[ERROR] Sample image not found: {sample_path}")
    sys.exit(1)

img = cv2.imread(sample_path)
h, w, c = img.shape
print(f"[INFO] Loaded sample image: {sample_path} ({w}x{h}, {c} channels)")

try:
    print(f"[INFO] Sending test inference request to Roboflow hosted model: {model_id}...")
    result = client.infer(sample_path, model_id=model_id)
    print("[SUCCESS] Roboflow API response received successfully!")
    print(f"[INFO] Response keys: {list(result.keys()) if isinstance(result, dict) else type(result)}")
    
    predictions = result.get("predictions", []) if isinstance(result, dict) else []
    print(f"[INFO] Found {len(predictions)} prediction(s):")
    
    for i, pred in enumerate(predictions):
        cls = pred.get("class")
        conf = pred.get("confidence")
        x = pred.get("x")
        y = pred.get("y")
        width = pred.get("width")
        height = pred.get("height")
        print(f"  - Prediction {i+1}: class='{cls}', confidence={conf:.4f}, bbox=[x={x}, y={y}, w={width}, h={height}]")

    if len(predictions) == 0:
        print("[FAIL] Zero predictions returned.")
        sys.exit(2)
        
    print("[PASSED] Real Roboflow inference check passed successfully!")

except Exception as e:
    print(f"[ERROR] Roboflow API call failed: {e}")
    sys.exit(1)

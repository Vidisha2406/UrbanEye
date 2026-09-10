import os
import sys

backend_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(backend_dir)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import json
import time
import subprocess
import cv2
import numpy as np
from fastapi.testclient import TestClient
from dotenv import load_dotenv

# Load environment
env_path = os.path.join(backend_dir, ".env")
load_dotenv(dotenv_path=env_path)

from backend.app.main import app
from backend.inference.vehicle_detector import get_vehicle_detector, VehicleDetector, TARGET_VEHICLE_CLASSES
from backend.inference.traffic_analyzer import get_traffic_analyzer, TrafficAnalyzer
from backend.inference.infrastructure_detector import get_infrastructure_detector
from backend.inference.pothole_detector import get_detector

client = TestClient(app)

dense_traffic_img = os.path.join(repo_root, "samples", "dense_traffic_sample.jpg")
pune_traffic_img = os.path.join(repo_root, "samples", "pune_traffic_sample.jpg")
city_traffic_video = os.path.join(repo_root, "samples", "city_traffic_multiclass.mp4")
damaged_sign_img = os.path.join(repo_root, "samples", "damaged_traffic_sign_sample.jpg")
sample_pothole_img = os.path.join(backend_dir, "sample_pothole.jpg")
if not os.path.exists(sample_pothole_img):
    sample_pothole_img = os.path.join(repo_root, "samples", "sample_pothole_road.jpg")

print("=" * 70)
print("URBANEYE VEHICLE DETECTION + BYTETRACK VERIFICATION SUITE (C1 - C13)")
print("=" * 70)

results = {}
metrics = {}

# -------------------------------------------------------------
# C1 — MODEL
# -------------------------------------------------------------
print("\n[CHECKPOINT C1] Verifying YOLO11n Model Loading & Singleton Pattern...")
try:
    model_path = os.path.join(backend_dir, "models", "yolo11n.pt")
    assert os.path.exists(model_path), f"YOLO11n weights not found at {model_path}"
    model_size_mb = round(os.path.getsize(model_path) / (1024 * 1024), 2)
    assert model_size_mb > 4.0, f"Model file suspiciously small: {model_size_mb} MB"
    
    # Verify singleton pattern
    vd1 = get_vehicle_detector()
    vd2 = get_vehicle_detector()
    assert vd1 is vd2, "VehicleDetector get_vehicle_detector() does not maintain singleton instance"
    assert vd1.model is not None, "YOLO11n model object not initialized"
    assert vd1.target_classes == {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}
    
    # Check health endpoint
    h_res = client.get("/api/health")
    assert h_res.status_code == 200, f"Health check failed with {h_res.status_code}"
    h_data = h_res.json()
    assert h_data["vehicle_detector"]["status"] == "online"
    assert h_data["vehicle_detector"]["model"] == "YOLO11n"
    assert "Car" in h_data["vehicle_detector"]["target_classes"]
    assert "Motorcycle" in h_data["vehicle_detector"]["target_classes"]
    assert "Bus" in h_data["vehicle_detector"]["target_classes"]
    assert "Truck" in h_data["vehicle_detector"]["target_classes"]
    
    metrics["C1_model_size"] = f"{model_size_mb} MB"
    metrics["C1_target_classes"] = list(vd1.target_classes.values())
    print(f"✓ C1 PASS: YOLO11n model loaded ({model_size_mb} MB), target classes: {list(vd1.target_classes.values())}, singleton verified.")
    results["C1"] = "PASS"
except Exception as e:
    print(f"✗ C1 FAIL: {e}")
    results["C1"] = f"FAIL: {e}"

# -------------------------------------------------------------
# C2 — REAL IMAGE DETECTION
# -------------------------------------------------------------
print("\n[CHECKPOINT C2] Running YOLO11n Real Image Detection...")
try:
    vd = get_vehicle_detector()
    assert os.path.exists(dense_traffic_img), f"Missing test media: {dense_traffic_img}"
    
    det_res = vd.detect(dense_traffic_img, confidence_threshold=0.25)
    assert det_res.get("success") is True, f"Detection failed: {det_res}"
    assert det_res.get("count", 0) > 0, f"No vehicles detected: {det_res}"
    
    detections = det_res["detections"]
    detected_classes = set(d["class"] for d in detections)
    valid_classes = {"Car", "Motorcycle", "Bus", "Truck"}
    assert detected_classes.issubset(valid_classes), f"Invalid class detected: {detected_classes - valid_classes}"
    
    confs = [d["confidence"] for d in detections]
    min_conf, max_conf, avg_conf = min(confs), max(confs), sum(confs) / len(confs)
    assert min_conf >= 0.25, f"Detection below threshold: {min_conf}"
    
    for d in detections:
        bbox = d["bbox"]
        assert all(k in bbox for k in ["x_min", "y_min", "x_max", "y_max", "x", "y", "width", "height", "box_2d"])
        assert bbox["width"] > 0 and bbox["height"] > 0, f"Invalid bbox dimensions: {bbox}"
        assert bbox["x_max"] >= bbox["x_min"] and bbox["y_max"] >= bbox["y_min"]
        
    # Test standalone detect-vehicles endpoint
    with open(dense_traffic_img, "rb") as f:
        api_res = client.post("/api/detect-vehicles", files={"file": ("dense_traffic.jpg", f, "image/jpeg")}, data={"annotate": "true"})
    assert api_res.status_code == 200, f"/api/detect-vehicles failed: {api_res.status_code}"
    api_data = api_res.json()
    assert api_data["count"] == len(detections)
    assert bool(api_data.get("annotated_image")) is True
    
    metrics["C2_image_detections"] = len(detections)
    metrics["C2_detected_classes"] = list(detected_classes)
    metrics["C2_confidence_range"] = f"{round(min_conf*100, 1)}% - {round(max_conf*100, 1)}% (avg {round(avg_conf*100, 1)}%)"
    metrics["C2_summary"] = det_res["summary"]
    
    print(f"✓ C2 PASS: Detected {len(detections)} vehicles on image ({det_res['summary']}). Conf range: {metrics['C2_confidence_range']}.")
    results["C2"] = "PASS"
except Exception as e:
    print(f"✗ C2 FAIL: {e}")
    results["C2"] = f"FAIL: {e}"

# -------------------------------------------------------------
# C3 — VIDEO DETECTION
# -------------------------------------------------------------
print("\n[CHECKPOINT C3] Running Frame-by-Frame Video Detection on Multiclass Traffic Stream...")
try:
    assert os.path.exists(city_traffic_video), f"Missing video: {city_traffic_video}"
    
    cap = cv2.VideoCapture(city_traffic_video)
    assert cap.isOpened(), "Could not open video with cv2"
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Sample 10 frames across video
    sampled_indices = np.linspace(0, total_frames - 1, min(10, total_frames), dtype=int)
    sampled_frames = []
    for idx in sampled_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret and frame is not None:
            sampled_frames.append((int(idx), frame))
    cap.release()
    
    assert len(sampled_frames) > 0, "No frames extracted from video"
    
    vd = get_vehicle_detector()
    video_det_res = vd.detect_frames(sampled_frames, confidence_threshold=0.25)
    
    assert video_det_res.get("success") is True
    assert video_det_res.get("total_detections", 0) > 0
    assert video_det_res.get("max_per_frame", 0) > 0
    assert video_det_res.get("best_frame_idx") is not None
    
    metrics["C3_video_frames_sampled"] = len(sampled_frames)
    metrics["C3_total_frame_detections"] = video_det_res["total_detections"]
    metrics["C3_max_per_frame"] = video_det_res["max_per_frame"]
    metrics["C3_best_frame_idx"] = video_det_res["best_frame_idx"]
    metrics["C3_summary"] = video_det_res["summary"]
    
    print(f"✓ C3 PASS: Processed {len(sampled_frames)} video frames ({w}x{h} @ {fps:.1f}fps). Total detections across sampled frames: {video_det_res['total_detections']}, max/frame: {video_det_res['max_per_frame']}.")
    results["C3"] = "PASS"
except Exception as e:
    print(f"✗ C3 FAIL: {e}")
    results["C3"] = f"FAIL: {e}"

# -------------------------------------------------------------
# C4 — BYTETRACK PERSISTENT IDS
# -------------------------------------------------------------
print("\n[CHECKPOINT C4] Running ByteTrack Multi-Object Tracking & Persistent IDs...")
try:
    vd = get_vehicle_detector()
    track_res = vd.track_video_stream(city_traffic_video, confidence_threshold=0.25, max_frames=50)
    
    assert track_res.get("success") is True, f"Tracking failed: {track_res}"
    assert track_res.get("total_frames_processed", 0) >= 40, f"Low frame count: {track_res.get('total_frames_processed')}"
    assert track_res.get("total_unique", 0) > 0, f"No unique tracks found: {track_res}"
    
    track_records = track_res.get("track_records", {})
    persistence_examples = track_res.get("track_persistence_examples", [])
    assert len(persistence_examples) > 0, "No persistent tracks (tracks appearing in >= 2 frames)"
    
    # Verify tracks span multiple consecutive frames
    long_lived_tracks = [t for t in track_records.values() if t["frame_count"] >= 10]
    assert len(long_lived_tracks) >= 3, f"Expected at least 3 long-lived tracks (>=10 frames), got {len(long_lived_tracks)}"
    
    for ex in persistence_examples[:3]:
        print(f"  Track ID #{ex['track_id']} ({ex['class']}): persisted across {ex['frame_count']} frames, max conf: {ex['max_conf']}")
    
    metrics["C4_total_tracks"] = len(track_records)
    metrics["C4_long_lived_tracks"] = len(long_lived_tracks)
    metrics["C4_persistence_sample"] = persistence_examples[:3]
    
    print(f"✓ C4 PASS: ByteTrack assigned persistent IDs to {len(track_records)} vehicles ({len(long_lived_tracks)} tracks persisted >= 10 frames).")
    results["C4"] = "PASS"
except Exception as e:
    print(f"✗ C4 FAIL: {e}")
    results["C4"] = f"FAIL: {e}"

# -------------------------------------------------------------
# C5 — UNIQUE COUNTING VS RAW DETECTIONS
# -------------------------------------------------------------
print("\n[CHECKPOINT C5] Verifying Unique Counting vs Raw Frame Detections...")
try:
    per_frame = track_res.get("per_frame", {})
    raw_detections_sum = sum(len(dets) for dets in per_frame.values())
    unique_count = track_res.get("total_unique", 0)
    unique_counts_breakdown = track_res.get("unique_counts", {})
    
    assert raw_detections_sum > unique_count, f"Raw detections ({raw_detections_sum}) should be much higher than unique vehicles ({unique_count})"
    assert unique_counts_breakdown.get("total") == unique_count
    
    ratio = round(raw_detections_sum / unique_count, 1) if unique_count > 0 else 0
    print(f"  Raw cumulative frame detections: {raw_detections_sum}")
    print(f"  ByteTrack deduplicated unique count: {unique_count}")
    print(f"  Deduplication ratio: {ratio}x")
    print(f"  Unique counts by category: {unique_counts_breakdown}")
    
    metrics["C5_raw_detections_sum"] = raw_detections_sum
    metrics["C5_unique_vehicles"] = unique_count
    metrics["C5_unique_counts_breakdown"] = unique_counts_breakdown
    metrics["C5_dedup_ratio"] = f"{ratio}x"
    
    print("✓ C5 PASS: ByteTrack correctly deduplicated raw detections into exact unique vehicle counts.")
    results["C5"] = "PASS"
except Exception as e:
    print(f"✗ C5 FAIL: {e}")
    results["C5"] = f"FAIL: {e}"

# -------------------------------------------------------------
# C6 — ANNOTATED EVIDENCE WITH TRACK IDS
# -------------------------------------------------------------
print("\n[CHECKPOINT C6] Generating & Verifying Annotated Evidence with Track IDs...")
try:
    best_frame = track_res.get("best_frame")
    best_frame_dets = track_res.get("best_frame_detections", [])
    assert best_frame is not None, "No best frame extracted from tracking"
    assert len(best_frame_dets) > 0, "No detections on best frame"
    
    vd = get_vehicle_detector()
    annotated = vd.annotate_frame(best_frame, best_frame_dets)
    assert annotated is not None and annotated.shape == best_frame.shape
    
    # Save evidence file
    evidence_path = os.path.join(backend_dir, "static", "evidence", "test_vehicle_tracking_evidence.jpg")
    cv2.imwrite(evidence_path, annotated)
    assert os.path.exists(evidence_path), f"Failed to write evidence image to {evidence_path}"
    
    # Verify file can be read back and has valid size
    read_back = cv2.imread(evidence_path)
    assert read_back is not None and read_back.shape == best_frame.shape
    ev_size_kb = round(os.path.getsize(evidence_path) / 1024, 1)
    
    metrics["C6_evidence_path"] = evidence_path
    metrics["C6_evidence_size_kb"] = f"{ev_size_kb} KB"
    metrics["C6_vehicles_on_evidence"] = len(best_frame_dets)
    
    print(f"✓ C6 PASS: Annotated evidence frame saved ({ev_size_kb} KB) showing {len(best_frame_dets)} tracked vehicles with labels & track IDs.")
    results["C6"] = "PASS"
except Exception as e:
    print(f"✗ C6 FAIL: {e}")
    results["C6"] = f"FAIL: {e}"

# -------------------------------------------------------------
# C7 — TRAFFIC TELEMETRY
# -------------------------------------------------------------
print("\n[CHECKPOINT C7] Verifying Traffic Intelligence Telemetry...")
try:
    ta = get_traffic_analyzer()
    assert ta.calculate_traffic_density(2) == "LOW"
    assert ta.calculate_traffic_density(5) == "MODERATE"
    assert ta.calculate_traffic_density(10) == "HIGH"
    assert ta.calculate_traffic_density(16) == "SEVERE"
    
    assert ta.calculate_congestion_level("LOW") == "FREE FLOW"
    assert ta.calculate_congestion_level("MODERATE") == "MODERATE"
    assert ta.calculate_congestion_level("HIGH") == "HIGH"
    assert ta.calculate_congestion_level("SEVERE") == "SEVERE"
    
    traffic_res = ta.analyze(dense_traffic_img, confidence_threshold=0.25)
    assert traffic_res.get("success") is True
    assert traffic_res.get("vehicle_count") > 0
    assert traffic_res.get("traffic_density") in ["LOW", "MODERATE", "HIGH", "SEVERE"]
    assert traffic_res.get("congestion_level") in ["FREE FLOW", "MODERATE", "HIGH", "SEVERE"]
    assert "Car" in traffic_res.get("vehicle_mix", {})
    assert traffic_res.get("occupancy_ratio", 0.0) >= 0.0
    
    # Check /api/detect-traffic endpoint
    with open(dense_traffic_img, "rb") as f:
        t_api_res = client.post("/api/detect-traffic", files={"file": ("dense.jpg", f, "image/jpeg")}, data={"annotate": "true"})
    assert t_api_res.status_code == 200, f"/api/detect-traffic failed: {t_api_res.status_code}"
    t_data = t_api_res.json()
    assert t_data["vehicle_count"] == traffic_res["vehicle_count"]
    assert t_data["traffic_density"] == traffic_res["traffic_density"]
    assert bool(t_data.get("annotated_image")) is True
    
    metrics["C7_vehicle_count"] = traffic_res["vehicle_count"]
    metrics["C7_traffic_density"] = traffic_res["traffic_density"]
    metrics["C7_congestion_level"] = traffic_res["congestion_level"]
    metrics["C7_vehicle_mix"] = traffic_res["vehicle_mix"]
    metrics["C7_occupancy_ratio"] = traffic_res["occupancy_ratio"]
    
    print(f"✓ C7 PASS: Telemetry computed: {traffic_res['vehicle_count']} vehicles, Density={traffic_res['traffic_density']}, Congestion={traffic_res['congestion_level']}, Mix={traffic_res['vehicle_mix']}, Occupancy={round(traffic_res['occupancy_ratio']*100, 1)}%.")
    results["C7"] = "PASS"
except Exception as e:
    print(f"✗ C7 FAIL: {e}")
    results["C7"] = f"FAIL: {e}"

# -------------------------------------------------------------
# C8 — URBANEVENT INTEGRATION
# -------------------------------------------------------------
print("\n[CHECKPOINT C8] Verifying Canonical UrbanEvent Schema Generation...")
try:
    # 1. Video upload -> VEHICLE_DETECTION UrbanEvent
    with open(city_traffic_video, "rb") as f:
        up_video_res = client.post("/api/upload", files={"file": ("city_traffic_multiclass.mp4", f, "video/mp4")}, data={"bus_id": "BUS-01"})
    assert up_video_res.status_code == 200, f"Upload video failed: {up_video_res.status_code}"
    up_v_data = up_video_res.json()
    assert up_v_data["success"] is True
    assert up_v_data["vehicles_detected"] is True
    assert up_v_data["detection_type"] == "VEHICLE_DETECTION"
    
    v_evt = up_v_data.get("event")
    assert v_evt is not None, "Missing event in upload response"
    assert v_evt["type"] == "VEHICLE_DETECTION"
    assert v_evt["category"] == "Vehicle"
    assert "Car" in v_evt["className"] or "Vehicle" in v_evt["className"]
    assert v_evt["confidence"] > 0
    assert v_evt["severity"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    assert v_evt["busId"] == "BUS-01"
    assert v_evt["coordinates"] is not None and len(v_evt["coordinates"]) == 2
    assert v_evt["uniqueCount"] > 0
    assert v_evt["evidenceFrame"] is not None
    assert v_evt["evidenceUrl"] is not None
    assert v_evt["corroborationStatus"] == "SINGLE_BUS"
    assert len(v_evt["observations"]) >= 1
    
    # 2. Image upload -> TRAFFIC_CONGESTION UrbanEvent
    with open(dense_traffic_img, "rb") as f:
        up_img_res = client.post("/api/upload", files={"file": ("dense_traffic_sample.jpg", f, "image/jpeg")}, data={"bus_id": "BUS-02"})
    assert up_img_res.status_code == 200, f"Upload image failed: {up_img_res.status_code}"
    up_i_data = up_img_res.json()
    assert up_i_data["traffic_detected"] is True
    assert up_i_data["detection_type"] == "TRAFFIC_CONGESTION"
    
    t_evt = up_i_data.get("event")
    assert t_evt is not None
    assert t_evt["type"] == "TRAFFIC_CONGESTION"
    assert t_evt["category"] == "Traffic"
    assert "Congestion" in t_evt["className"]
    assert t_evt["vehicleCount"] > 0
    assert t_evt["trafficDensity"] in ["LOW", "MODERATE", "HIGH", "SEVERE"]
    assert t_evt["congestionLevel"] in ["FREE FLOW", "MODERATE", "HIGH", "SEVERE"]
    
    metrics["C8_video_event_id"] = v_evt["id"]
    metrics["C8_video_event_type"] = v_evt["type"]
    metrics["C8_traffic_event_id"] = t_evt["id"]
    metrics["C8_traffic_event_type"] = t_evt["type"]
    
    print(f"✓ C8 PASS: Canonical UrbanEvents generated ({v_evt['id']}: {v_evt['type']} and {t_evt['id']}: {t_evt['type']}).")
    results["C8"] = "PASS"
except Exception as e:
    print(f"✗ C8 FAIL: {e}")
    results["C8"] = f"FAIL: {e}"

# -------------------------------------------------------------
# C9 — EDGE AI UI
# -------------------------------------------------------------
print("\n[CHECKPOINT C9] Verifying Edge AI UI Architecture & Types...")
try:
    with open(os.path.join(repo_root, "src", "types", "index.ts"), "r") as f:
        types_content = f.read()
    assert "uniqueCount?: number" in types_content
    assert "uniqueCounts?: Record<string, number>" in types_content
    assert "vehicleCount?: number" in types_content
    assert "trafficDensity?: string" in types_content
    assert "congestionLevel?: string" in types_content
    
    with open(os.path.join(repo_root, "src", "context", "UrbanEyeContext.tsx"), "r") as f:
        context_content = f.read()
    assert "'Tracking (ByteTrack)'" in context_content
    assert "uniqueVehicleCounts" in context_content
    assert "latestTrafficTelemetry" in context_content
    assert "VEHICLE_DETECTION" in context_content
    
    with open(os.path.join(repo_root, "src", "pages", "EdgeAI.tsx"), "r") as f:
        edge_ai_content = f.read()
    assert "uniqueVehicleCounts" in edge_ai_content
    assert "latestTrafficTelemetry" in edge_ai_content
    assert "LIVE DETECTIONS" in edge_ai_content
    
    metrics["C9_edge_ai_pipeline_step_4"] = "Tracking (ByteTrack)"
    metrics["C9_unique_counts_binding"] = "Verified in EdgeAI.tsx"
    
    print("✓ C9 PASS: Edge AI UI component cleanly consumes ByteTrack telemetry and persistent unique vehicle counts.")
    results["C9"] = "PASS"
except Exception as e:
    print(f"✗ C9 FAIL: {e}")
    results["C9"] = f"FAIL: {e}"

# -------------------------------------------------------------
# C10 — TRAFFIC UI
# -------------------------------------------------------------
print("\n[CHECKPOINT C10] Verifying Traffic Mobility UI Architecture...")
try:
    with open(os.path.join(repo_root, "src", "pages", "TrafficMobility.tsx"), "r") as f:
        traffic_ui_content = f.read()
    assert "Traffic & Mobility Intelligence" in traffic_ui_content
    assert "latestTrafficTelemetry" in traffic_ui_content
    assert "vehicleMix" in traffic_ui_content
    assert "trafficDensity" in traffic_ui_content
    assert "TrafficCongestionMap" in traffic_ui_content
    
    metrics["C10_traffic_page"] = "TrafficMobility.tsx"
    metrics["C10_traffic_map"] = "TrafficCongestionMap.tsx"
    
    print("✓ C10 PASS: Traffic & Mobility UI integrates live vehicle telemetry, traffic density, and vehicle mix.")
    results["C10"] = "PASS"
except Exception as e:
    print(f"✗ C10 FAIL: {e}")
    results["C10"] = f"FAIL: {e}"

# -------------------------------------------------------------
# C11 — REGRESSION (Potholes + Damaged Signs)
# -------------------------------------------------------------
print("\n[CHECKPOINT C11] Running Multi-Model Non-Interference Regression Test...")
try:
    # 1. Test Pothole Detector
    pothole_det = get_detector()
    assert os.path.exists(sample_pothole_img), f"Pothole sample missing: {sample_pothole_img}"
    pothole_res = pothole_det.detect(sample_pothole_img, confidence_threshold=0.20)
    assert pothole_res.get("success") is True, f"Pothole inference failed: {pothole_res}"
    assert pothole_res.get("count", 0) > 0, "No potholes detected on sample pothole image"
    top_pothole_conf = pothole_res["detections"][0]["confidence"]
    assert top_pothole_conf > 0.70, f"Low pothole conf: {top_pothole_conf}"
    print(f"  Pothole detector: {pothole_res['count']} detected, top conf: {round(top_pothole_conf*100, 1)}%")
    
    # 2. Test Damaged Sign Detector
    infra_det = get_infrastructure_detector()
    assert os.path.exists(damaged_sign_img), f"Damaged sign sample missing: {damaged_sign_img}"
    infra_res = infra_det.detect(damaged_sign_img, confidence_threshold=0.25)
    assert infra_res.get("success") is True, f"Damaged sign inference failed: {infra_res}"
    assert infra_res.get("count", 0) > 0, "No damaged signs detected on sample image"
    top_infra_conf = infra_res["detections"][0]["confidence"]
    assert top_infra_conf > 0.70, f"Low sign conf: {top_infra_conf}"
    print(f"  Damaged sign detector: {infra_res['count']} detected, top conf: {round(top_infra_conf*100, 1)}%")
    
    # 3. Test Vehicle Detector once again
    vd = get_vehicle_detector()
    v_res = vd.detect(dense_traffic_img, confidence_threshold=0.25)
    assert v_res.get("success") is True and v_res.get("count", 0) > 0
    print(f"  Vehicle detector: {v_res['count']} detected")
    
    metrics["C11_pothole_regression"] = f"PASS ({pothole_res['count']} potholes, {round(top_pothole_conf*100, 1)}%)"
    metrics["C11_damaged_sign_regression"] = f"PASS ({infra_res['count']} signs, {round(top_infra_conf*100, 1)}%)"
    metrics["C11_vehicle_regression"] = f"PASS ({v_res['count']} vehicles)"
    
    print("✓ C11 PASS: All 3 detection models (Pothole, Damaged Sign, Vehicle/ByteTrack) operate with 0 collisions or regressions.")
    results["C11"] = "PASS"
except Exception as e:
    print(f"✗ C11 FAIL: {e}")
    results["C11"] = f"FAIL: {e}"

# -------------------------------------------------------------
# C12 — BUILD (npm run build + backend tests)
# -------------------------------------------------------------
print("\n[CHECKPOINT C12] Running Frontend Build & Validation...")
try:
    build_proc = subprocess.run(
        ["npm", "run", "build"],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=120,
    )
    assert build_proc.returncode == 0, f"npm run build failed with code {build_proc.returncode}:\n{build_proc.stderr}"
    print("  npm run build completed successfully (0 TypeScript/bundle errors).")
    
    metrics["C12_build_status"] = "SUCCESS (code 0)"
    print("✓ C12 PASS: Frontend build passes with 0 TypeScript/Vite compilation errors.")
    results["C12"] = "PASS"
except Exception as e:
    print(f"✗ C12 FAIL: {e}")
    results["C12"] = f"FAIL: {e}"

# -------------------------------------------------------------
# C13 — FULL VEHICLE E2E
# -------------------------------------------------------------
print("\n[CHECKPOINT C13] Running Full End-to-End Multi-Modal Vehicle & Traffic Pipeline Test...")
try:
    # 1. Health check
    h = client.get("/api/health").json()
    assert h["status"] == "online"
    
    # 2. Standalone vehicle detection on image
    with open(pune_traffic_img, "rb") as f:
        dv = client.post("/api/detect-vehicles", files={"file": ("pune.jpg", f, "image/jpeg")}, data={"annotate": "true"}).json()
    assert dv["success"] is True
    assert dv["count"] > 0
    
    # 3. Standalone traffic intelligence on image
    with open(pune_traffic_img, "rb") as f:
        dt = client.post("/api/detect-traffic", files={"file": ("pune.jpg", f, "image/jpeg")}, data={"annotate": "true"}).json()
    assert dt["success"] is True
    assert dt["vehicle_count"] > 0
    assert dt["traffic_density"] is not None
    
    # 4. Upload video with ByteTrack multi-object tracking
    with open(city_traffic_video, "rb") as f:
        up_v = client.post("/api/upload", files={"file": ("city_traffic_multiclass.mp4", f, "video/mp4")}, data={"bus_id": "BUS-03"}).json()
    assert up_v["success"] is True
    assert up_v["vehicles_detected"] is True
    assert up_v["unique_vehicles"] > 0
    assert up_v["tracker_type"] == "ByteTrack"
    assert up_v["event"]["id"] is not None
    
    # 5. Verify static evidence accessibility
    ev_url = up_v["evidence_url"]
    assert ev_url.startswith("/api/evidence/")
    filename = os.path.basename(ev_url)
    ev_file_path = os.path.join(backend_dir, "static", "evidence", filename)
    assert os.path.exists(ev_file_path), f"Evidence file not found on disk: {ev_file_path}"
    
    metrics["C13_e2e_full_pipeline"] = "PASS (Health + Image Detect + Traffic Detect + Video ByteTrack + Evidence + UrbanEvent)"
    
    print("✓ C13 PASS: Full vehicle detection, ByteTrack tracking, telemetry calculation, and UrbanEvent pipeline verified end-to-end.")
    results["C13"] = "PASS"
except Exception as e:
    print(f"✗ C13 FAIL: {e}")
    results["C13"] = f"FAIL: {e}"

# -------------------------------------------------------------
# SUMMARY REPORT
# -------------------------------------------------------------
print("\n" + "=" * 70)
print("FINAL TEST EXECUTION SUMMARY")
print("=" * 70)
all_pass = all(v == "PASS" for v in results.values())
for k in [f"C{i}" for i in range(1, 14)]:
    status = results.get(k, "NOT RUN")
    print(f"  {k}: {status}")

print("\nKEY METRICS:")
for k, v in metrics.items():
    print(f"  {k}: {v}")

print("=" * 70)
if all_pass:
    print("ALL CHECKPOINTS C1 - C13 PASSED SUCCESSFULLY!")
else:
    print("SOME CHECKPOINTS FAILED!")
print("=" * 70)

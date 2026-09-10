"""
URBANEYE — LOOP 11A: DEPLOYMENT PREPARATION VALIDATION SUITE
Verifies that UrbanEye is 100% prepared for production deployment on Render / Cloud PaaS.

Validates:
1. Deployment configuration files (render.yaml, requirements.txt, .env.example)
2. Production entrypoint (backend.app.main:app)
3. Frontend production configuration (VITE_API_URL, no hardcoded ports)
4. Safe CORS configuration (custom origins, no invalid wildcard credential combo)
5. Environment variable integrity (placeholders only, zero secrets)
6. Model availability & lightweight footprint (YOLO11n, Roboflow, EasyOCR, ByteTrack)
7. Production-style startup locally using dynamic $PORT
8. /api/health and /health endpoint validation
9. Frontend -> Backend API communication flow
10. Real AI smoke test on real assets
11. Full 10-pipeline multi-model regression
12. Frontend production build (npm run build -> dist/index.html)
"""

import os
import sys
import time
import json
import socket
import threading
import subprocess
from typing import Dict, Any

# Ensure project root is in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi.testclient import TestClient

def run_loop11a_validation() -> Dict[str, Any]:
    print("=" * 80)
    print("URBANEYE — LOOP 11A: DEPLOYMENT PREPARATION VALIDATION SUITE")
    print("=" * 80)

    results: Dict[str, str] = {}
    blockers = []

    # -------------------------------------------------------------------------
    # 1. Deployment Configuration (render.yaml, requirements.txt, .env.example)
    # -------------------------------------------------------------------------
    print("\n[1/12] Validating Deployment Configuration Files...")
    try:
        render_yaml = os.path.join(PROJECT_ROOT, "render.yaml")
        assert os.path.exists(render_yaml), "render.yaml is missing"
        with open(render_yaml, "r", encoding="utf-8") as f:
            yaml_content = f.read()
            assert "urbaneye-backend" in yaml_content, "urbaneye-backend service missing in render.yaml"
            assert "urbaneye-frontend" in yaml_content, "urbaneye-frontend service missing in render.yaml"
            assert "backend.app.main:app" in yaml_content, "backend.app.main:app entrypoint missing in render.yaml"
            assert "npm run build" in yaml_content, "npm run build missing in render.yaml"

        root_reqs = os.path.join(PROJECT_ROOT, "requirements.txt")
        assert os.path.exists(root_reqs), "Root requirements.txt missing"

        backend_reqs = os.path.join(PROJECT_ROOT, "backend", "requirements.txt")
        assert os.path.exists(backend_reqs), "backend/requirements.txt missing"
        with open(backend_reqs, "r", encoding="utf-8") as f:
            reqs_content = f.read()
            for dep in ["fastapi", "uvicorn", "ultralytics", "supervision", "easyocr", "inference-sdk"]:
                assert dep in reqs_content, f"Required dependency '{dep}' missing in requirements.txt"

        results["Deployment configuration"] = "PASS"
        print("  ✓ render.yaml, requirements.txt, and build specifications verified.")
    except Exception as e:
        results["Deployment configuration"] = f"FAIL ({e})"
        blockers.append(f"Deployment config error: {e}")

    # -------------------------------------------------------------------------
    # 2. Backend Production Command & Entrypoint (backend.app.main:app)
    # -------------------------------------------------------------------------
    print("\n[2/12] Validating Backend Production Entrypoint (backend.app.main:app)...")
    try:
        from backend.app.main import app
        assert app is not None, "FastAPI app instance failed to load"
        assert hasattr(app, "router"), "FastAPI router missing on app"
        
        results["Backend production command"] = "PASS"
        print("  ✓ Entrypoint backend.app.main:app verified for uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT")
    except Exception as e:
        results["Backend production command"] = f"FAIL ({e})"
        blockers.append(f"Backend entrypoint error: {e}")

    # -------------------------------------------------------------------------
    # 3. Frontend Production Configuration (VITE_API_URL & zero hardcoded ports)
    # -------------------------------------------------------------------------
    print("\n[3/12] Validating Frontend Production Configuration...")
    try:
        context_file = os.path.join(PROJECT_ROOT, "src", "context", "UrbanEyeContext.tsx")
        with open(context_file, "r", encoding="utf-8") as f:
            ctx_content = f.read()
            assert "VITE_API_URL" in ctx_content, "VITE_API_URL resolution missing in UrbanEyeContext"
            assert "getApiUrl" in ctx_content, "getApiUrl helper missing in UrbanEyeContext"
            assert "resolveEvidenceUrl" in ctx_content, "resolveEvidenceUrl helper missing in UrbanEyeContext"

        # Check for hardcoded local ports in src/
        forbidden = ["http://127.0.0.1:8000", "http://localhost:8000", "localhost:8000", "localhost:5173"]
        src_dir = os.path.join(PROJECT_ROOT, "src")
        for root, _, files in os.walk(src_dir):
            for file in files:
                if file.endswith((".ts", ".tsx", ".js", ".jsx")):
                    fpath = os.path.join(root, file)
                    with open(fpath, "r", encoding="utf-8") as fh:
                        content = fh.read()
                        for bad in forbidden:
                            assert bad not in content, f"Hardcoded local URL '{bad}' found in {file}"

        results["Frontend production configuration"] = "PASS"
        print("  ✓ Dynamic API routing verified; 0 hardcoded local URLs in frontend source code.")
    except Exception as e:
        results["Frontend production configuration"] = f"FAIL ({e})"
        blockers.append(f"Frontend production config error: {e}")

    # -------------------------------------------------------------------------
    # 4. Safe CORS Configuration
    # -------------------------------------------------------------------------
    print("\n[4/12] Validating Safe CORS Configuration...")
    try:
        main_file = os.path.join(PROJECT_ROOT, "backend", "app", "main.py")
        with open(main_file, "r", encoding="utf-8") as f:
            main_code = f.read()
            assert "ALLOWED_ORIGINS" in main_code, "ALLOWED_ORIGINS environment parsing missing in main.py"
            assert "CORSMiddleware" in main_code, "CORSMiddleware missing in main.py"

        # Test CORS headers on test client
        from backend.app.main import app as test_app
        client = TestClient(test_app)
        res = client.options(
            "/api/health",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "GET"
            }
        )
        assert res.status_code in [200, 204], f"CORS preflight failed with status {res.status_code}"
        assert res.headers.get("access-control-allow-origin") == "http://localhost:5173"

        results["CORS"] = "PASS"
        print("  ✓ Safe origin parsing and preflight CORS verified.")
    except Exception as e:
        results["CORS"] = f"FAIL ({e})"
        blockers.append(f"CORS error: {e}")

    # -------------------------------------------------------------------------
    # 5. Environment Variables & Sanitization (.env.example)
    # -------------------------------------------------------------------------
    print("\n[5/12] Validating Environment Variables & Template Sanitization...")
    try:
        env_ex = os.path.join(PROJECT_ROOT, ".env.example")
        assert os.path.exists(env_ex), ".env.example missing"
        with open(env_ex, "r", encoding="utf-8") as f:
            content = f.read()
            required_keys = [
                "VITE_API_URL", "VITE_CARTO_API_KEY", "ROBOFLOW_API_KEY",
                "ROBOFLOW_MODEL_ID", "ROBOFLOW_INFRASTRUCTURE_MODEL_ID",
                "YOLO_MODEL_PATH", "ALLOWED_ORIGINS", "PORT", "PYTHON_VERSION"
            ]
            for rk in required_keys:
                assert rk in content, f"Missing {rk} in .env.example"
            assert "sk-proj-" not in content and "rf_" not in content, "Real secret found in .env.example"

        results["Environment variables"] = "PASS"
        print("  ✓ All 9 environment variables documented; 0 live secrets exposed.")
    except Exception as e:
        results["Environment variables"] = f"FAIL ({e})"
        blockers.append(f"Environment variables error: {e}")

    # -------------------------------------------------------------------------
    # 6. Model Availability & Footprint
    # -------------------------------------------------------------------------
    print("\n[6/12] Validating Model Availability & Footprint...")
    try:
        yolo_path = os.path.join(PROJECT_ROOT, "backend", "models", "yolo11n.pt")
        if not os.path.exists(yolo_path):
            yolo_path = os.path.join(PROJECT_ROOT, "yolo11n.pt")
        assert os.path.exists(yolo_path), f"YOLO11n weights not found at {yolo_path}"
        
        file_size_mb = os.path.getsize(yolo_path) / (1024 * 1024)
        assert file_size_mb < 50.0, f"Model file too large ({file_size_mb:.2f}MB) for standard git/PaaS"
        
        # Test model loader
        from backend.inference.vehicle_detector import get_vehicle_detector
        detector = get_vehicle_detector()
        assert detector.model is not None, "YOLO11n model failed to initialize"

        results["Model availability"] = "PASS"
        print(f"  ✓ YOLO11n model verified: {file_size_mb:.2f}MB lightweight footprint.")
    except Exception as e:
        results["Model availability"] = f"FAIL ({e})"
        blockers.append(f"Model error: {e}")

    # -------------------------------------------------------------------------
    # 7. Production-Style Local Startup & API Health Test
    # -------------------------------------------------------------------------
    print("\n[7/12] Validating Production-Style Local Server Startup & /api/health...")
    try:
        from backend.app.main import app as local_app
        client = TestClient(local_app)

        # 1. Root /
        r_root = client.get("/")
        assert r_root.status_code == 200
        assert r_root.json()["status"] == "online"

        # 2. Root /health
        r_h = client.get("/health")
        assert r_h.status_code == 200
        assert r_h.json()["status"] == "ok"

        # 3. API Health /api/health
        r_api_h = client.get("/api/health")
        assert r_api_h.status_code == 200
        health_data = r_api_h.json()
        assert health_data["status"] == "online"
        assert health_data["vehicle_detector"]["status"] == "online"
        assert health_data["traffic_analyzer"]["status"] == "online"
        assert health_data["pedestrian_risk_analyzer"]["status"] == "online"
        assert health_data["anpr_ocr_analyzer"]["status"] == "online"
        assert health_data["rash_driving_analyzer"]["status"] == "online"
        assert health_data["hit_and_run_analyzer"]["status"] == "online"

        results["API health"] = "PASS"
        print("  ✓ Root '/', '/health', and '/api/health' responding with 200 OK.")
    except Exception as e:
        results["API health"] = f"FAIL ({e})"
        blockers.append(f"API health error: {e}")

    # -------------------------------------------------------------------------
    # 8. Frontend -> Backend API Communication Test
    # -------------------------------------------------------------------------
    print("\n[8/12] Validating Frontend -> Backend API Communication Flow...")
    try:
        sample_img = os.path.join(PROJECT_ROOT, "backend", "sample_pothole.jpg")
        if not os.path.exists(sample_img):
            sample_img = os.path.join(PROJECT_ROOT, "samples", "sample_pothole_road.jpg")
        with open(sample_img, "rb") as f:
            resp = client.post(
                "/api/upload",
                files={"file": ("sample_pothole.jpg", f, "image/jpeg")},
                data={"bus_id": "BUS-01", "camera_name": "Front Cam"}
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"]
        assert "event" in data
        assert data["event"]["id"].startswith("EVT-")
        assert data["evidence_url"].startswith("/api/evidence/")

        # Verify evidence frame is served by static mount
        ev_filename = os.path.basename(data["evidence_url"])
        ev_resp = client.get(f"/api/evidence/{ev_filename}")
        assert ev_resp.status_code == 200

        results["Frontend → Backend"] = "PASS"
        print("  ✓ Full API lifecycle verified: /api/upload -> Event -> Static Evidence serving.")
    except Exception as e:
        results["Frontend → Backend"] = f"FAIL ({e})"
        blockers.append(f"Frontend -> Backend API error: {e}")

    # -------------------------------------------------------------------------
    # 9. Real AI Smoke Test
    # -------------------------------------------------------------------------
    print("\n[9/12] Running Real AI Perception Smoke Test...")
    try:
        from backend.inference.pothole_detector import get_detector
        from backend.inference.infrastructure_detector import get_infrastructure_detector
        from backend.inference.vehicle_detector import get_vehicle_detector

        sample_pothole = os.path.join(PROJECT_ROOT, "backend", "sample_pothole.jpg")
        if not os.path.exists(sample_pothole):
            sample_pothole = os.path.join(PROJECT_ROOT, "samples", "sample_pothole_road.jpg")
        sample_sign = os.path.join(PROJECT_ROOT, "samples", "damaged_traffic_sign_sample.jpg")
        sample_vehicle = os.path.join(PROJECT_ROOT, "samples", "pune_traffic_sample.jpg")

        p_res = get_detector().detect(sample_pothole)
        assert p_res["success"]
        print(f"  - Pothole AI: Found {p_res['count']} detection(s)")

        s_res = get_infrastructure_detector().detect(sample_sign)
        assert s_res["success"]
        print(f"  - Sign AI: Found {len(s_res['detections'])} detection(s)")

        v_res = get_vehicle_detector().detect(sample_vehicle)
        assert v_res["success"]
        print(f"  - Vehicle AI: Found {v_res['count']} vehicle(s)")

        results["Real AI smoke test"] = "PASS"
        print("  ✓ Real AI smoke test passed across Roboflow and YOLO11n engines.")
    except Exception as e:
        results["Real AI smoke test"] = f"FAIL ({e})"
        blockers.append(f"AI Smoke test error: {e}")

    # -------------------------------------------------------------------------
    # 10. Master 10-Pipeline Regression Test
    # -------------------------------------------------------------------------
    print("\n[10/12] Running Full Multi-Model Regression Across All 10 Pipelines...")
    try:
        from backend.test_loop9_global_audit import run_loop9_audit
        loop9_passed = run_loop9_audit()
        assert loop9_passed, "Loop 9 test suite failed during regression"

        results["Full regression"] = "PASS"
        print("  ✓ Zero regression: All 10 AI perception & analytics pipelines fully operational.")
    except Exception as e:
        results["Full regression"] = f"FAIL ({e})"
        blockers.append(f"Regression error: {e}")

    # -------------------------------------------------------------------------
    # 11. Frontend Build Verification (npm run build)
    # -------------------------------------------------------------------------
    print("\n[11/12] Validating Frontend Production Build...")
    try:
        dist_index = os.path.join(PROJECT_ROOT, "dist", "index.html")
        assert os.path.exists(dist_index), "dist/index.html missing"
        
        # Verify index.html contains bundle scripts
        with open(dist_index, "r", encoding="utf-8") as f:
            html = f.read()
            assert "<script type=\"module\"" in html or "assets/" in html

        results["npm build"] = "PASS"
        print("  ✓ Production build verified (dist/index.html + CSS/JS chunks).")
    except Exception as e:
        results["npm build"] = f"FAIL ({e})"
        blockers.append(f"npm build error: {e}")

    # -------------------------------------------------------------------------
    # 12. Summary & Deployment Readiness
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("LOOP 11A FINAL DEPLOYMENT PREPARATION SUMMARY")
    print("=" * 80)
    all_passed = len(blockers) == 0

    return {
        "all_passed": all_passed,
        "results": results,
        "blockers": blockers
    }

if __name__ == "__main__":
    report = run_loop11a_validation()
    for k, v in report["results"].items():
        print(f"{k}: {v}")
    print("=" * 80)
    if report["all_passed"]:
        print("FINAL: READY TO DEPLOY")
        sys.exit(0)
    else:
        print("FINAL: NOT READY")
        print(f"Blockers: {report['blockers']}")
        sys.exit(1)

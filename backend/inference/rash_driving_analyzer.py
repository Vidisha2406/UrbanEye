import os
import cv2
import time
import math
import numpy as np
from typing import Union, List, Dict, Any, Optional, Tuple
from collections import defaultdict, Counter
import supervision as sv

from backend.inference.vehicle_detector import get_vehicle_detector, VehicleDetector, TARGET_VEHICLE_CLASSES

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(CURRENT_DIR)

# Status Constants
STATUS_RASH_DRIVING = "RASH DRIVING"
STATUS_SUSPICIOUS = "SUSPICIOUS MOTION"
STATUS_STABLE = "STABLE MOTION"

# Deterministic Rule Constants
RULE_ABRUPT_DECELERATION = "ABRUPT_DECELERATION"
RULE_ABRUPT_ACCELERATION = "ABRUPT_ACCELERATION"
RULE_LATERAL_SWERVE = "LATERAL_SWERVE_DEVIATION"
RULE_ZIG_ZAG = "ZIG_ZAG_MANEUVER"
RULE_TRAJECTORY_INSTABILITY = "TRAJECTORY_INSTABILITY"

# Rule Human-Readable Labels
RULE_LABELS = {
    RULE_ABRUPT_DECELERATION: "Abrupt Deceleration / Sudden Braking",
    RULE_ABRUPT_ACCELERATION: "Abrupt Acceleration Surge",
    RULE_LATERAL_SWERVE: "Abrupt Lateral Swerve Deviation",
    RULE_ZIG_ZAG: "Erratic Zig-Zag Lane Deviation",
    RULE_TRAJECTORY_INSTABILITY: "High Trajectory Instability",
}

class RashDrivingAnalyzer:
    """
    Real Rash Driving Analyzer for UrbanEye Edge AI.
    
    Architecture:
    1. YOLO11n Vehicle Detection (Car, Motorcycle, Bus, Truck)
    2. ByteTrack Persistent Multi-Object Tracking & Track History
    3. Deterministic Kinematic & Trajectory Motion Analysis:
       - Frame-to-frame displacement vectors
       - Relative speed proxy in image space (pixels/frame)
       - Relative acceleration & deceleration proxies (pixels/frame^2)
       - Heading angle & lateral angular deviation
       - Lateral direction reversals (zig-zag metric)
       - Trajectory curvature variance / instability
    4. Transparent Deterministic Rule Engine:
       - Evaluates kinematic metrics against calibrated thresholds
       - Scores risk from 0 to 100
       - Classifies into 'RASH DRIVING', 'SUSPICIOUS MOTION', or 'STABLE MOTION'
    5. Annotated Telemetry Evidence Generation:
       - Historical trajectory trails
       - Vehicle bounding boxes & ByteTrack ID tags
       - Infraction rule pills & motion telemetry HUD
    """

    def __init__(
        self,
        vehicle_detector: Optional[VehicleDetector] = None,
        min_track_length: int = 3,
        swerve_angle_threshold: float = 30.0,
        decel_threshold: float = -7.0,
        accel_threshold: float = 8.0,
        zigzag_min_reversals: int = 2,
        instability_threshold: float = 22.0,
    ):
        self.vehicle_detector = vehicle_detector or get_vehicle_detector()
        self.min_track_length = min_track_length
        self.swerve_angle_threshold = swerve_angle_threshold
        self.decel_threshold = decel_threshold
        self.accel_threshold = accel_threshold
        self.zigzag_min_reversals = zigzag_min_reversals
        self.instability_threshold = instability_threshold
        self.model_name = "YOLO11n + ByteTrack Kinematic Analyzer"

    def analyze_trajectory(
        self,
        track_id: int,
        vehicle_class: str,
        positions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Performs deterministic kinematic and trajectory analysis on a single vehicle track history.
        positions: list of dicts with keys: 'frame_idx', 'x', 'y', 'width', 'height', 'bbox'
        """
        def normalize_pos(p):
            p_copy = dict(p)
            bbox = p_copy.get("bbox", {})
            if "x" not in p_copy:
                p_copy["x"] = float(bbox.get("x", (bbox.get("x_min", 0.0) + bbox.get("x_max", 0.0)) / 2.0))
            if "y" not in p_copy:
                p_copy["y"] = float(bbox.get("y", (bbox.get("y_min", 0.0) + bbox.get("y_max", 0.0)) / 2.0))
            if "width" not in p_copy:
                p_copy["width"] = float(bbox.get("width", bbox.get("x_max", 0.0) - bbox.get("x_min", 0.0)))
            if "height" not in p_copy:
                p_copy["height"] = float(bbox.get("height", bbox.get("y_max", 0.0) - bbox.get("y_min", 0.0)))
            return p_copy

        positions = [normalize_pos(p) for p in positions]

        n_points = len(positions)
        if n_points < self.min_track_length:
            return {
                "track_id": track_id,
                "vehicle_class": vehicle_class,
                "track_length": n_points,
                "is_sufficient_length": False,
                "status": STATUS_STABLE,
                "is_rash_driving": False,
                "is_suspicious": False,
                "risk_score": 0,
                "severity": "LOW",
                "triggered_rules": [],
                "rule_details": [],
                "motion_metrics": {
                    "avg_speed_proxy": 0.0,
                    "max_speed_proxy": 0.0,
                    "max_decel_proxy": 0.0,
                    "max_accel_proxy": 0.0,
                    "max_swerve_deg": 0.0,
                    "zigzag_reversals": 0,
                    "trajectory_instability_std": 0.0,
                },
                "trajectory_points": positions,
                "latest_bbox": positions[-1]["bbox"] if positions else {},
                "latest_frame_idx": positions[-1]["frame_idx"] if positions else 0,
            }

        # 1. Compute displacements & speed proxies
        speeds: List[float] = []
        headings: List[float] = []
        dx_list: List[float] = []
        dy_list: List[float] = []

        for i in range(1, n_points):
            p_prev = positions[i - 1]
            p_curr = positions[i]

            dx = p_curr["x"] - p_prev["x"]
            dy = p_curr["y"] - p_prev["y"]
            disp = math.hypot(dx, dy)

            # Frame gap normalization if frames were dropped
            frame_gap = max(1, p_curr["frame_idx"] - p_prev["frame_idx"])
            speed_proxy = disp / float(frame_gap)

            speeds.append(speed_proxy)
            dx_list.append(dx / float(frame_gap))
            dy_list.append(dy / float(frame_gap))

            # Heading angle in degrees (-180 to 180)
            if disp > 1.5:  # filter sub-pixel jitter
                heading = math.degrees(math.atan2(dy, dx))
                headings.append(heading)
            elif headings:
                headings.append(headings[-1])
            else:
                headings.append(0.0)

        # 2. Compute accelerations / decelerations
        accelerations: List[float] = []
        for i in range(1, len(speeds)):
            dv = speeds[i] - speeds[i - 1]
            accelerations.append(dv)

        # 3. Compute lateral angular deviations
        swerve_angles: List[float] = []
        for i in range(1, len(headings)):
            dh = abs(headings[i] - headings[i - 1])
            # normalize angle difference to [0, 180]
            if dh > 180.0:
                dh = 360.0 - dh
            swerve_angles.append(dh)

        # 4. Compute zig-zag reversals (sign changes in lateral velocity dx)
        zigzag_reversals = 0
        sig_dx = [x for x in dx_list if abs(x) > 2.0]
        for i in range(1, len(sig_dx)):
            if (sig_dx[i] > 0 and sig_dx[i - 1] < 0) or (sig_dx[i] < 0 and sig_dx[i - 1] > 0):
                zigzag_reversals += 1

        # 5. Compute summary metrics
        avg_speed = float(np.mean(speeds)) if speeds else 0.0
        max_speed = float(np.max(speeds)) if speeds else 0.0
        min_accel = float(np.min(accelerations)) if accelerations else 0.0  # most negative = max deceleration
        max_accel = float(np.max(accelerations)) if accelerations else 0.0
        max_swerve = float(np.max(swerve_angles)) if swerve_angles else 0.0
        instability_std = float(np.std(swerve_angles)) if len(swerve_angles) >= 2 else 0.0

        # 6. Deterministic Rule Evaluations
        triggered_rules: List[str] = []
        rule_details: List[str] = []
        risk_score = 10  # baseline normal movement score

        # Rule 1: Abrupt Deceleration (Sudden Braking)
        if min_accel <= self.decel_threshold:
            triggered_rules.append(RULE_ABRUPT_DECELERATION)
            rule_details.append(f"Sudden deceleration of {abs(round(min_accel, 2))} px/f² detected")
            risk_score += 30

        # Rule 2: Abrupt Acceleration Surge
        if max_accel >= self.accel_threshold:
            triggered_rules.append(RULE_ABRUPT_ACCELERATION)
            rule_details.append(f"Rapid acceleration surge of {round(max_accel, 2)} px/f² detected")
            risk_score += 25

        # Rule 3: Lateral Swerve / Abrupt Direction Change
        if max_swerve >= self.swerve_angle_threshold:
            triggered_rules.append(RULE_LATERAL_SWERVE)
            rule_details.append(f"Sharp lateral heading deviation of {round(max_swerve, 1)}° detected")
            risk_score += 35

        # Rule 4: Zig-Zag Lane Maneuver
        if zigzag_reversals >= self.zigzag_min_reversals:
            triggered_rules.append(RULE_ZIG_ZAG)
            rule_details.append(f"Erratic lateral zig-zagging ({zigzag_reversals} reversals) detected")
            risk_score += 40

        # Rule 5: Trajectory Instability
        if instability_std >= self.instability_threshold:
            triggered_rules.append(RULE_TRAJECTORY_INSTABILITY)
            rule_details.append(f"High trajectory path instability (std: {round(instability_std, 1)}°)")
            risk_score += 25

        # Clamp risk score to [0, 100]
        risk_score = max(0, min(100, risk_score))

        # Classify status
        if risk_score >= 70 or (len(triggered_rules) >= 2 and risk_score >= 60):
            status = STATUS_RASH_DRIVING
            severity = "CRITICAL" if risk_score >= 85 else "HIGH"
        elif risk_score >= 40 or len(triggered_rules) >= 1:
            status = STATUS_SUSPICIOUS
            severity = "MEDIUM"
        else:
            status = STATUS_STABLE
            severity = "LOW"

        return {
            "track_id": track_id,
            "vehicle_class": vehicle_class,
            "track_length": n_points,
            "is_sufficient_length": True,
            "status": status,
            "is_rash_driving": status == STATUS_RASH_DRIVING,
            "is_suspicious": status in [STATUS_RASH_DRIVING, STATUS_SUSPICIOUS],
            "risk_score": risk_score,
            "severity": severity,
            "triggered_rules": triggered_rules,
            "rule_details": rule_details,
            "motion_metrics": {
                "avg_speed_proxy": round(avg_speed, 2),
                "max_speed_proxy": round(max_speed, 2),
                "max_decel_proxy": round(min_accel, 2),
                "max_accel_proxy": round(max_accel, 2),
                "max_swerve_deg": round(max_swerve, 1),
                "zigzag_reversals": zigzag_reversals,
                "trajectory_instability_std": round(instability_std, 1),
            },
            "trajectory_points": positions,
            "latest_bbox": positions[-1]["bbox"] if positions else {},
            "latest_frame_idx": positions[-1]["frame_idx"] if positions else 0,
        }

    def analyze_video_stream(
        self,
        video_path: str,
        confidence_threshold: float = 0.25,
        max_frames: int = 150,
        sample_interval: int = 1,
    ) -> Dict[str, Any]:
        """
        Executes end-to-end YOLO11n + ByteTrack + Kinematic Rash Driving Analysis on a video file.
        """
        start_time = time.time()

        if not os.path.exists(video_path):
            return {
                "success": False,
                "error": f"Video path not found: {video_path}",
                "total_frames": 0,
                "processed_frames": 0,
                "unique_tracks": 0,
                "rash_candidates": [],
                "suspicious_tracks": [],
                "best_candidate": None,
                "processing_time_sec": 0.0,
            }

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {
                "success": False,
                "error": f"Failed to open video file: {video_path}",
                "total_frames": 0,
                "processed_frames": 0,
                "unique_tracks": 0,
                "rash_candidates": [],
                "suspicious_tracks": [],
                "best_candidate": None,
                "processing_time_sec": 0.0,
            }

        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        tracker = sv.ByteTrack(
            track_activation_threshold=confidence_threshold,
            lost_track_buffer=30,
            minimum_matching_threshold=0.8,
        )

        frame_idx = 0
        processed_count = 0
        total_detections_count = 0
        track_histories: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        track_classes: Dict[int, Counter] = defaultdict(Counter)
        track_latest_frame: Dict[int, np.ndarray] = {}

        while cap.isOpened() and frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            if frame_idx % sample_interval == 0:
                processed_count += 1
                h, w = frame.shape[:2]

                # Run YOLO11n vehicle detection
                results = self.vehicle_detector.model(
                    frame,
                    classes=list(self.vehicle_detector.target_classes.keys()),
                    conf=confidence_threshold,
                    verbose=False,
                )[0]

                detections = sv.Detections.from_ultralytics(results)
                total_detections_count += len(detections)

                # Update ByteTrack tracker
                tracked = tracker.update_with_detections(detections)

                for xyxy, mask, conf, class_id, tracker_id, data in tracked:
                    cls_id = int(class_id)
                    cls_name = self.vehicle_detector.target_classes.get(
                        cls_id, results.names.get(cls_id, "Vehicle").capitalize()
                    )
                    tid = int(tracker_id)
                    c_conf = float(conf)

                    x_min = max(0.0, min(float(w), float(xyxy[0])))
                    y_min = max(0.0, min(float(h), float(xyxy[1])))
                    x_max = max(0.0, min(float(w), float(xyxy[2])))
                    y_max = max(0.0, min(float(h), float(xyxy[3])))
                    box_w = max(0.0, x_max - x_min)
                    box_h = max(0.0, y_max - y_min)
                    center_x = x_min + box_w / 2.0
                    center_y = y_min + box_h / 2.0

                    track_classes[tid][cls_name] += 1
                    track_histories[tid].append({
                        "frame_idx": frame_idx,
                        "x": center_x,
                        "y": center_y,
                        "width": box_w,
                        "height": box_h,
                        "confidence": c_conf,
                        "bbox": {
                            "x_min": round(x_min, 2),
                            "y_min": round(y_min, 2),
                            "x_max": round(x_max, 2),
                            "y_max": round(y_max, 2),
                            "width": round(box_w, 2),
                            "height": round(box_h, 2),
                        },
                    })
                    track_latest_frame[tid] = frame.copy()

            frame_idx += 1

        cap.release()

        # Perform kinematic analysis on all accumulated tracks
        analyzed_tracks: List[Dict[str, Any]] = []
        rash_candidates: List[Dict[str, Any]] = []
        suspicious_tracks: List[Dict[str, Any]] = []
        best_candidate: Optional[Dict[str, Any]] = None
        max_risk = -1

        for tid, positions in track_histories.items():
            majority_class = track_classes[tid].most_common(1)[0][0] if track_classes[tid] else "Vehicle"
            analysis = self.analyze_trajectory(tid, majority_class, positions)
            analyzed_tracks.append(analysis)

            if analysis["is_rash_driving"]:
                rash_candidates.append(analysis)
            elif analysis["is_suspicious"]:
                suspicious_tracks.append(analysis)

            if analysis["risk_score"] > max_risk:
                max_risk = analysis["risk_score"]
                best_candidate = analysis

        # Sort candidates by risk score descending
        rash_candidates.sort(key=lambda c: c["risk_score"], reverse=True)
        suspicious_tracks.sort(key=lambda c: c["risk_score"], reverse=True)

        elapsed = time.time() - start_time

        return {
            "success": True,
            "total_frames": total_video_frames,
            "processed_frames": processed_count,
            "vehicle_detections_count": total_detections_count,
            "unique_tracks_count": len(track_histories),
            "tracks_analyzed_count": len([t for t in analyzed_tracks if t["is_sufficient_length"]]),
            "rash_candidates_count": len(rash_candidates),
            "suspicious_tracks_count": len(suspicious_tracks),
            "rash_candidates": rash_candidates,
            "suspicious_tracks": suspicious_tracks,
            "all_analyzed_tracks": analyzed_tracks,
            "best_candidate": best_candidate,
            "strongest_risk_score": max_risk if max_risk >= 0 else 0,
            "processing_time_sec": round(elapsed, 3),
            "track_frames_cache": track_latest_frame,
        }

    def annotate_frame(
        self,
        frame: np.ndarray,
        candidate: Dict[str, Any],
        bus_id: str = "BUS-01",
        route_name: str = "Route 17",
        all_active_tracks: Optional[List[Dict[str, Any]]] = None
    ) -> np.ndarray:
        """
        Generates a high-contrast annotated evidence frame displaying:
        - Vehicle bounding box (vibrant crimson/rose for rash, cyan for background)
        - Historical trajectory motion trail with connected waypoints
        - Kinematic infraction rule pills & risk severity badge
        - Top Telemetry HUD banner with bus ID & route
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX

        # 1. Draw other active background tracks if provided
        if all_active_tracks:
            for trk in all_active_tracks:
                if trk.get("track_id") == candidate.get("track_id"):
                    continue
                bbox = trk.get("latest_bbox", {})
                if bbox:
                    bx1 = max(0, int(bbox.get("x_min", 0)))
                    by1 = max(0, int(bbox.get("y_min", 0)))
                    bx2 = min(w, int(bbox.get("x_max", w)))
                    by2 = min(h, int(bbox.get("y_max", h)))
                    cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (200, 220, 0), 1)

        # 2. Draw historical trajectory trail for candidate
        points = candidate.get("trajectory_points", [])
        if len(points) >= 2:
            for i in range(1, len(points)):
                pt1 = (int(points[i - 1]["x"]), int(points[i - 1]["y"]))
                pt2 = (int(points[i]["x"]), int(points[i]["y"]))
                
                # Trajectory color gradient: Cyan to Rose
                progress = i / float(len(points))
                color = (
                    int(255 * progress),
                    int(100 * (1 - progress)),
                    int(255 * (1 - progress))
                )
                cv2.line(annotated, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)
                cv2.circle(annotated, pt2, 4, (0, 255, 255), -1)

        # 3. Draw Candidate Vehicle Bounding Box
        v_bbox = candidate.get("latest_bbox", {})
        tid = candidate.get("track_id", 1)
        v_class = candidate.get("vehicle_class", "Vehicle")
        risk_score = candidate.get("risk_score", 0)
        severity = candidate.get("severity", "LOW")
        rules = candidate.get("triggered_rules", [])

        if v_bbox:
            vx1 = max(0, int(v_bbox.get("x_min", 0)))
            vy1 = max(0, int(v_bbox.get("y_min", 0)))
            vx2 = min(w, int(v_bbox.get("x_max", w)))
            vy2 = min(h, int(v_bbox.get("y_max", h)))

            box_color = (0, 0, 240) if risk_score >= 70 else (0, 165, 255)  # Crimson for Rash, Amber for Suspicious

            cv2.rectangle(annotated, (vx1, vy1), (vx2, vy2), box_color, 3)

            # Corner accents
            c_len = min(16, max(4, (vx2 - vx1) // 4))
            cv2.line(annotated, (vx1, vy1), (vx1 + c_len, vy1), (255, 255, 255), 2)
            cv2.line(annotated, (vx1, vy1), (vx1, vy1 + c_len), (255, 255, 255), 2)
            cv2.line(annotated, (vx2, vy1), (vx2 - c_len, vy1), (255, 255, 255), 2)
            cv2.line(annotated, (vx2, vy1), (vx2, vy1 + c_len), (255, 255, 255), 2)

            # Target ID Tag Header
            header_str = f"TARGET: {v_class} #{tid} | RISK: {risk_score}/100 ({severity})"
            (htw, hth), _ = cv2.getTextSize(header_str, font, 0.52, 2)
            tag_y = max(hth + 10, vy1 - 8) if vy1 > hth + 12 else min(h - 5, vy2 + hth + 14)
            cv2.rectangle(annotated, (vx1, tag_y - hth - 6), (vx1 + htw + 12, tag_y + 4), (0, 0, 0), -1)
            cv2.rectangle(annotated, (vx1, tag_y - hth - 6), (vx1 + htw + 12, tag_y + 4), box_color, 1)
            cv2.putText(annotated, header_str, (vx1 + 6, tag_y - 2), font, 0.52, (255, 255, 255), 2, cv2.LINE_AA)

            # Infraction Rule Tag Below Box
            if rules:
                rule_str = " | ".join([RULE_LABELS.get(r, r) for r in rules[:2]])
                (rtw, rth), _ = cv2.getTextSize(rule_str, font, 0.45, 1)
                rule_y = min(h - 10, vy2 + rth + 14)
                cv2.rectangle(annotated, (vx1, rule_y - rth - 4), (vx1 + rtw + 10, rule_y + 4), (10, 10, 30), -1)
                cv2.putText(annotated, rule_str, (vx1 + 5, rule_y), font, 0.45, (0, 200, 255), 1, cv2.LINE_AA)

        # 4. Top Telemetry & HUD Banner
        banner_h = 44
        cv2.rectangle(annotated, (0, 0), (w, banner_h), (15, 22, 24), -1)
        cv2.line(annotated, (0, banner_h), (w, banner_h), (0, 0, 255) if risk_score >= 70 else (0, 165, 255), 2)

        # Status Circle Indicator
        indicator_color = (0, 0, 255) if risk_score >= 70 else (0, 200, 255)
        cv2.circle(annotated, (20, banner_h // 2), 6, indicator_color, -1)

        hud_title = f"URBANEYE RASH DRIVING INTELLIGENCE | Fleet: {bus_id} ({route_name})"
        hud_details = f"Target: {v_class} #{tid} | Risk: {risk_score}/100 ({severity}) | Status: {candidate.get('status', STATUS_STABLE)} | Rules: {len(rules)}"

        cv2.putText(annotated, hud_title, (36, 18), font, 0.45, (200, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(annotated, hud_details, (36, 36), font, 0.46, (0, 240, 255), 1, cv2.LINE_AA)

        return annotated


# Global Singleton instance
_RASH_ANALYZER_INSTANCE: Optional[RashDrivingAnalyzer] = None

def get_rash_driving_analyzer() -> RashDrivingAnalyzer:
    global _RASH_ANALYZER_INSTANCE
    if _RASH_ANALYZER_INSTANCE is None:
        _RASH_ANALYZER_INSTANCE = RashDrivingAnalyzer()
    return _RASH_ANALYZER_INSTANCE

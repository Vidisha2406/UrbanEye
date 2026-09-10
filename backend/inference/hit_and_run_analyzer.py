import os
import cv2
import time
import math
import numpy as np
from typing import Union, List, Dict, Any, Optional, Tuple
from collections import defaultdict, Counter
import supervision as sv

from backend.inference.vehicle_detector import get_vehicle_detector, VehicleDetector, TARGET_VEHICLE_CLASSES
from backend.inference.anpr_ocr_analyzer import get_anpr_ocr_analyzer, ANPROCRAnalyzer, STATE_RELIABLE_PLATE, STATE_OCR_UNCERTAIN, STATE_NO_RELIABLE_PLATE

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(CURRENT_DIR)

# Status Constants
STATUS_NO_INCIDENT = "NO INCIDENT"
STATUS_POSSIBLE_INCIDENT = "POSSIBLE INCIDENT CANDIDATE"
STATUS_HIT_AND_RUN_CANDIDATE = "HIT & RUN CANDIDATE"
STATUS_CONFIRMED = "CONFIRMED INCIDENT"

# Deterministic Rule Constants
RULE_SPATIAL_PROXIMITY_COLLISION_PROXY = "SPATIAL_PROXIMITY_COLLISION_PROXY"
RULE_ABRUPT_KINEMATIC_CHANGE = "ABRUPT_KINEMATIC_CHANGE"
RULE_POST_INTERACTION_DEPARTURE = "POST_INTERACTION_DEPARTURE"
RULE_INTERIOR_SCENE_DISAPPEARANCE = "INTERIOR_SCENE_DISAPPEARANCE"
RULE_ASYMMETRIC_FLEEING = "ASYMMETRIC_FLEEING"
RULE_NORMAL_CAMERA_EXIT = "NORMAL_CAMERA_EXIT"

# Human-Readable Rule Labels
RULE_LABELS = {
    RULE_SPATIAL_PROXIMITY_COLLISION_PROXY: "Spatial Proximity / Collision Proxy",
    RULE_ABRUPT_KINEMATIC_CHANGE: "Abrupt Kinematic Velocity / Swerve Disruption",
    RULE_POST_INTERACTION_DEPARTURE: "Post-Interaction High-Speed Departure",
    RULE_INTERIOR_SCENE_DISAPPEARANCE: "Interior Scene Track Disappearance",
    RULE_ASYMMETRIC_FLEEING: "Asymmetric Fleeing Post-Impact",
    RULE_NORMAL_CAMERA_EXIT: "Normal Camera Field Boundary Exit",
}


class HitAndRunAnalyzer:
    """
    Real Hit & Run Incident Analyzer for UrbanEye Edge AI.
    
    Architecture:
    1. YOLO11n Vehicle Detection (Car, Motorcycle, Bus, Truck)
    2. ByteTrack Persistent Multi-Object Tracking & Spatio-Temporal History
    3. Pairwise Spatio-Temporal Interaction Analysis:
       - Center-to-center Euclidean distances across concurrent frames
       - Normalized bounding box proximity ratio & bounding box overlap
       - Identification of closest interaction timestamp/frame (f_interact)
    4. Kinematic Anomaly & Trajectory Deviation Verification:
       - Abrupt deceleration spike or acceleration surge around f_interact
       - Lateral heading swerve / directional shift around f_interact
    5. Post-Interaction Spatio-Temporal Reasoning:
       - Differentiation between normal camera boundary exit vs interior scene disappearance
       - Asymmetric fleeing: suspect vehicle accelerates/departs while counterpart ceases tracking/stops
    6. Multi-Signal Deterministic Rule Engine:
       - Evaluates interaction against 5 transparent deterministic criteria
       - Computes incident risk score (0-100)
       - Classifies into 'NO INCIDENT', 'POSSIBLE INCIDENT CANDIDATE', or 'HIT & RUN CANDIDATE'
    7. Multi-Modal ANPR / OCR Correlation:
       - Associates verified OCR plate text with involved suspect track if available
    8. Annotated Telemetry Evidence HUD Generation:
       - Suspect vs impacted vehicle trajectory trails
       - Collision proxy reticle & interaction marker
       - HUD telemetry banner, rule pills, ANPR tag, and risk metrics
    """

    def __init__(
        self,
        vehicle_detector: Optional[VehicleDetector] = None,
        anpr_analyzer: Optional[ANPROCRAnalyzer] = None,
        proximity_threshold_px: float = 75.0,
        decel_spike_threshold: float = -5.0,
        accel_surge_threshold: float = 6.0,
        swerve_spike_threshold: float = 25.0,
        edge_margin_px: int = 40,
        min_track_length: int = 3,
    ):
        self.vehicle_detector = vehicle_detector or get_vehicle_detector()
        self._anpr_analyzer = anpr_analyzer
        self.proximity_threshold_px = proximity_threshold_px
        self.decel_spike_threshold = decel_spike_threshold
        self.accel_surge_threshold = accel_surge_threshold
        self.swerve_spike_threshold = swerve_spike_threshold
        self.edge_margin_px = edge_margin_px
        self.min_track_length = min_track_length
        self.model_name = "YOLO11n + ByteTrack Multi-Signal Incident Analyzer"

    @property
    def anpr_analyzer(self) -> ANPROCRAnalyzer:
        if self._anpr_analyzer is None:
            self._anpr_analyzer = get_anpr_ocr_analyzer()
        return self._anpr_analyzer

    def analyze_interaction(
        self,
        track_a: Dict[str, Any],
        track_b: Dict[str, Any],
        frame_width: int = 1280,
        frame_height: int = 720,
    ) -> Dict[str, Any]:
        """
        Analyzes the spatio-temporal interaction between two tracked vehicles.
        track_a, track_b: dicts containing:
          - 'track_id': int
          - 'vehicle_class': str
          - 'positions': list of dicts with 'frame_idx', 'x', 'y', 'width', 'height', 'bbox'
        """
        pos_a = track_a.get("positions", [])
        pos_b = track_b.get("positions", [])

        tid_a = track_a.get("track_id", 0)
        tid_b = track_b.get("track_id", 0)
        cls_a = track_a.get("vehicle_class", "Vehicle")
        cls_b = track_b.get("vehicle_class", "Vehicle")

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

        pos_a = [normalize_pos(p) for p in pos_a]
        pos_b = [normalize_pos(p) for p in pos_b]

        if len(pos_a) < self.min_track_length or len(pos_b) < self.min_track_length:
            return {
                "track_id_a": tid_a,
                "track_id_b": tid_b,
                "vehicle_class_a": cls_a,
                "vehicle_class_b": cls_b,
                "is_sufficient_length": False,
                "status": STATUS_NO_INCIDENT,
                "is_hit_and_run": False,
                "is_possible_incident": False,
                "risk_score": 0,
                "severity": "LOW",
                "triggered_rules": [],
                "rule_details": [],
                "interaction_metrics": {
                    "min_distance_px": 9999.0,
                    "min_proximity_ratio": 999.0,
                    "interaction_frame": None,
                    "max_decel_proxy": 0.0,
                    "max_accel_surge": 0.0,
                    "max_swerve_deg": 0.0,
                    "post_interaction_speed_a": 0.0,
                    "post_interaction_speed_b": 0.0,
                    "ended_at_boundary_a": False,
                    "ended_at_boundary_b": False,
                },
                "positions_a": pos_a,
                "positions_b": pos_b,
                "interaction_point": None,
                "anpr_correlation": None,
            }

        # Index positions by frame_idx
        map_a = {p["frame_idx"]: p for p in pos_a}
        map_b = {p["frame_idx"]: p for p in pos_b}

        common_frames = sorted(set(map_a.keys()) & set(map_b.keys()))

        # Compute displacements & speeds for both tracks
        def compute_kinematics(positions: List[Dict[str, Any]]):
            speeds = {}
            accelerations = {}
            headings = {}
            for i in range(1, len(positions)):
                p_prev = positions[i - 1]
                p_curr = positions[i]
                dx = p_curr["x"] - p_prev["x"]
                dy = p_curr["y"] - p_prev["y"]
                disp = math.hypot(dx, dy)
                gap = max(1, p_curr["frame_idx"] - p_prev["frame_idx"])
                spd = disp / float(gap)
                f_curr = p_curr["frame_idx"]
                speeds[f_curr] = spd
                if disp > 1.5:
                    headings[f_curr] = math.degrees(math.atan2(dy, dx))
                else:
                    headings[f_curr] = 0.0

            sorted_frames = sorted(speeds.keys())
            for i in range(1, len(sorted_frames)):
                f_prev = sorted_frames[i - 1]
                f_curr = sorted_frames[i]
                gap = max(1, f_curr - f_prev)
                acc = (speeds[f_curr] - speeds[f_prev]) / float(gap)
                accelerations[f_curr] = acc

            return speeds, accelerations, headings

        speeds_a, accs_a, heads_a = compute_kinematics(pos_a)
        speeds_b, accs_b, heads_b = compute_kinematics(pos_b)

        # 1. Proximity & Spatial Interaction
        min_dist = 9999.0
        min_prox_ratio = 999.0
        interaction_frame = None
        interaction_point = None

        if common_frames:
            for f in common_frames:
                pa = map_a[f]
                pb = map_b[f]
                dx = pa["x"] - pb["x"]
                dy = pa["y"] - pb["y"]
                dist = math.hypot(dx, dy)
                norm_size = max(float(pa.get("width", 50)), float(pa.get("height", 50)),
                                float(pb.get("width", 50)), float(pb.get("height", 50)), 1.0)
                prox_ratio = dist / norm_size

                if dist < min_dist:
                    min_dist = dist
                    min_prox_ratio = prox_ratio
                    interaction_frame = f
                    interaction_point = {
                        "x": round((pa["x"] + pb["x"]) / 2.0, 1),
                        "y": round((pa["y"] + pb["y"]) / 2.0, 1),
                        "frame_idx": f,
                        "distance_px": round(dist, 1),
                        "prox_ratio": round(prox_ratio, 2),
                    }
        else:
            # No concurrent frames — check closest endpoint proximity
            last_a = pos_a[-1]
            first_b = pos_b[0]
            dist = math.hypot(last_a["x"] - first_b["x"], last_a["y"] - first_b["y"])
            min_dist = dist
            min_prox_ratio = dist / 60.0

        # Check boundary exits for both vehicles
        def check_boundary_exit(p_last: Dict[str, Any], w: int, h: int, margin: int) -> bool:
            bbox = p_last.get("bbox", {})
            x_min = bbox.get("x_min", p_last.get("x", 0) - p_last.get("width", 0) / 2)
            y_min = bbox.get("y_min", p_last.get("y", 0) - p_last.get("height", 0) / 2)
            x_max = bbox.get("x_max", p_last.get("x", 0) + p_last.get("width", 0) / 2)
            y_max = bbox.get("y_max", p_last.get("y", 0) + p_last.get("height", 0) / 2)

            return (
                x_min <= margin or
                y_min <= margin or
                x_max >= (w - margin) or
                y_max >= (h - margin)
            )

        ended_at_boundary_a = check_boundary_exit(pos_a[-1], frame_width, frame_height, self.edge_margin_px)
        ended_at_boundary_b = check_boundary_exit(pos_b[-1], frame_width, frame_height, self.edge_margin_px)

        # 2. Kinematic Disturbances around interaction frame
        max_decel_a = min([accs_a[f] for f in accs_a], default=0.0)
        max_decel_b = min([accs_b[f] for f in accs_b], default=0.0)
        worst_decel = min(max_decel_a, max_decel_b)

        max_accel_a = max([accs_a[f] for f in accs_a], default=0.0)
        max_accel_b = max([accs_b[f] for f in accs_b], default=0.0)
        worst_accel = max(max_accel_a, max_accel_b)

        # Swerves
        def max_swerve(heads: Dict[int, float]) -> float:
            sorted_f = sorted(heads.keys())
            max_sw = 0.0
            for i in range(1, len(sorted_f)):
                diff = abs(heads[sorted_f[i]] - heads[sorted_f[i - 1]])
                if diff > 180:
                    diff = 360 - diff
                if diff > max_sw:
                    max_sw = diff
            return max_sw

        max_swerve_a = max_swerve(heads_a)
        max_swerve_b = max_swerve(heads_b)
        worst_swerve = max(max_swerve_a, max_swerve_b)

        # 3. Post-interaction Speeds
        post_speeds_a = []
        post_speeds_b = []
        pre_speeds_a = []
        pre_speeds_b = []
        if interaction_frame is not None:
            pre_speeds_a = [speeds_a[f] for f in speeds_a if f < interaction_frame]
            pre_speeds_b = [speeds_b[f] for f in speeds_b if f < interaction_frame]
            post_speeds_a = [speeds_a[f] for f in speeds_a if f >= interaction_frame]
            post_speeds_b = [speeds_b[f] for f in speeds_b if f >= interaction_frame]

        avg_pre_speed_a = float(np.mean(pre_speeds_a)) if pre_speeds_a else (speeds_a.get(pos_a[0]["frame_idx"], 0.0))
        avg_post_speed_a = float(np.mean(post_speeds_a)) if post_speeds_a else (speeds_a.get(pos_a[-1]["frame_idx"], 0.0))
        avg_post_speed_b = float(np.mean(post_speeds_b)) if post_speeds_b else (speeds_b.get(pos_b[-1]["frame_idx"], 0.0))

        # 4. Multi-Signal Deterministic Rule Evaluation
        triggered_rules: List[str] = []
        rule_details: List[str] = []
        risk_score = 10

        # Rule 1: Spatial Proximity Collision Proxy
        is_proximity_triggered = (min_dist <= self.proximity_threshold_px) or (min_prox_ratio <= 1.2)
        if is_proximity_triggered:
            triggered_rules.append(RULE_SPATIAL_PROXIMITY_COLLISION_PROXY)
            rule_details.append(
                f"Close Spatial Proximity / Collision Proxy: Min Center Distance {min_dist:.1f} px (Proximity Ratio: {min_prox_ratio:.2f})"
            )
            risk_score += 30

        # Rule 2: Abrupt Kinematic Change / Velocity Disruption
        is_kinematic_triggered = (
            worst_decel <= self.decel_spike_threshold or
            worst_accel >= self.accel_surge_threshold or
            worst_swerve >= self.swerve_spike_threshold
        )
        if is_kinematic_triggered and is_proximity_triggered:
            triggered_rules.append(RULE_ABRUPT_KINEMATIC_CHANGE)
            rule_details.append(
                f"Abrupt Kinematic Velocity Disruption: Max Decel {worst_decel:.1f} px/f², Accel Surge {worst_accel:.1f} px/f², Swerve {worst_swerve:.1f}°"
            )
            risk_score += 25

        # Rule 3: Post-Interaction Departure / Speed Surge
        is_post_departure = (
            (avg_post_speed_a >= 14.0) or
            (avg_post_speed_a >= 8.0 and avg_post_speed_a > avg_pre_speed_a * 1.25) or
            (max_accel_a >= 5.0 and avg_post_speed_a >= 8.0)
        )
        if is_post_departure and is_proximity_triggered:
            triggered_rules.append(RULE_POST_INTERACTION_DEPARTURE)
            rule_details.append(
                f"Post-Interaction High-Speed Departure: Suspect Track #{tid_a} departing at relative speed proxy {avg_post_speed_a:.1f} px/f (Pre-Speed: {avg_pre_speed_a:.1f} px/f)"
            )
            risk_score += 20

        # Rule 4: Interior Scene Disappearance
        last_frame_a = pos_a[-1]["frame_idx"]
        last_frame_b = pos_b[-1]["frame_idx"]
        disappeared_early = (last_frame_a >= last_frame_b + 2) or (interaction_frame is not None and last_frame_b <= interaction_frame + 1 and last_frame_a > last_frame_b)

        is_interior_disappearance = (not ended_at_boundary_b) and disappeared_early and (len(pos_b) >= self.min_track_length)
        if is_interior_disappearance and is_proximity_triggered:
            triggered_rules.append(RULE_INTERIOR_SCENE_DISAPPEARANCE)
            rule_details.append(
                f"Interior Scene Track Disappearance: Impacted Track #{tid_b} ceased tracking at frame {last_frame_b} within observable corridor ({pos_b[-1]['x']:.1f}, {pos_b[-1]['y']:.1f}), NOT at frame boundaries"
            )
            risk_score += 15
        elif ended_at_boundary_b:
            rule_details.append(
                f"Impacted Track #{tid_b} exited normally through camera field boundary (edge margin {self.edge_margin_px} px)"
            )

        # Rule 5: Asymmetric Fleeing Post-Impact
        is_asymmetric_fleeing = (
            is_proximity_triggered and
            (is_post_departure or max_accel_a >= 4.0) and
            (is_interior_disappearance or (last_frame_a >= last_frame_b + 2 and avg_post_speed_a > avg_post_speed_b * 1.5))
        )
        if is_asymmetric_fleeing:
            triggered_rules.append(RULE_ASYMMETRIC_FLEEING)
            rule_details.append(
                f"Asymmetric Fleeing Post-Impact: Track #{tid_a} fled while Track #{tid_b} remained stationary or ceased progression"
            )
            risk_score += 15

        risk_score = min(100, max(10, risk_score))

        # Classify Severity & Status
        if risk_score >= 80:
            severity = "CRITICAL"
        elif risk_score >= 60:
            severity = "HIGH"
        elif risk_score >= 40:
            severity = "MEDIUM"
        else:
            severity = "LOW"

        # Determine Status
        if (
            risk_score >= 70 and
            RULE_SPATIAL_PROXIMITY_COLLISION_PROXY in triggered_rules and
            RULE_ABRUPT_KINEMATIC_CHANGE in triggered_rules and
            (RULE_POST_INTERACTION_DEPARTURE in triggered_rules or RULE_INTERIOR_SCENE_DISAPPEARANCE in triggered_rules or RULE_ASYMMETRIC_FLEEING in triggered_rules) and
            len(triggered_rules) >= 3
        ):
            status = STATUS_HIT_AND_RUN_CANDIDATE
            is_hit_and_run = True
            is_possible = True
        elif risk_score >= 40 and is_proximity_triggered:
            status = STATUS_POSSIBLE_INCIDENT
            is_hit_and_run = False
            is_possible = True
        else:
            status = STATUS_NO_INCIDENT
            is_hit_and_run = False
            is_possible = False

        return {
            "track_id_a": tid_a,
            "track_id_b": tid_b,
            "vehicle_class_a": cls_a,
            "vehicle_class_b": cls_b,
            "is_sufficient_length": True,
            "status": status,
            "is_hit_and_run": is_hit_and_run,
            "is_possible_incident": is_possible,
            "risk_score": risk_score,
            "severity": severity,
            "triggered_rules": triggered_rules,
            "rule_details": rule_details,
            "interaction_metrics": {
                "min_distance_px": round(min_dist, 1),
                "min_proximity_ratio": round(min_prox_ratio, 2),
                "interaction_frame": interaction_frame,
                "max_decel_proxy": round(worst_decel, 2),
                "max_accel_surge": round(worst_accel, 2),
                "max_swerve_deg": round(worst_swerve, 1),
                "post_interaction_speed_a": round(avg_post_speed_a, 2),
                "post_interaction_speed_b": round(avg_post_speed_b, 2),
                "ended_at_boundary_a": ended_at_boundary_a,
                "ended_at_boundary_b": ended_at_boundary_b,
            },
            "positions_a": pos_a,
            "positions_b": pos_b,
            "interaction_point": interaction_point,
            "anpr_correlation": None,
        }

    def analyze_video_stream(
        self,
        video_input: str,
        confidence_threshold: float = 0.25,
        max_frames: int = 150,
    ) -> Dict[str, Any]:
        """
        Runs full end-to-end Hit & Run analysis on a real video stream.
        """
        start_time = time.time()

        if not os.path.exists(video_input):
            return {
                "success": False,
                "error": f"Video file not found: {video_input}",
                "total_frames": 0,
                "processed_frames": 0,
                "vehicle_detections_count": 0,
                "unique_tracks_count": 0,
                "tracks_analyzed_count": 0,
                "interaction_candidates_count": 0,
                "possible_incidents_count": 0,
                "hit_and_run_candidates_count": 0,
                "strongest_candidate": None,
                "risk_score": 0,
                "status": STATUS_NO_INCIDENT,
                "severity": "LOW",
                "all_analyzed_interactions": [],
                "processing_time_sec": 0.0,
            }

        cap = cv2.VideoCapture(video_input)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        cap.release()

        # 1. Run persistent ByteTrack multi-object tracking
        track_res = self.vehicle_detector.track_video_stream(
            video_input,
            confidence_threshold=confidence_threshold,
            max_frames=max_frames
        )

        if not track_res.get("success", False):
            return {
                "success": False,
                "error": track_res.get("error", "Tracking failed"),
                "total_frames": total_frames,
                "processed_frames": 0,
                "vehicle_detections_count": 0,
                "unique_tracks_count": 0,
                "tracks_analyzed_count": 0,
                "interaction_candidates_count": 0,
                "possible_incidents_count": 0,
                "hit_and_run_candidates_count": 0,
                "strongest_candidate": None,
                "risk_score": 0,
                "status": STATUS_NO_INCIDENT,
                "severity": "LOW",
                "all_analyzed_interactions": [],
                "processing_time_sec": round(time.time() - start_time, 3),
            }

        per_frame = track_res.get("per_frame", {})
        processed_frames = len(per_frame)
        best_frame = track_res.get("best_frame")
        best_frame_idx = track_res.get("best_frame_idx", 0)

        # 2. Reconstruct trajectory positions for each track ID
        track_positions: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        track_classes: Dict[int, Counter] = defaultdict(Counter)

        total_detections = 0
        for f_idx, detections in per_frame.items():
            total_detections += len(detections)
            for det in detections:
                tid = det["track_id"]
                cls_name = det["class"]
                bbox = det["bbox"]
                track_classes[tid][cls_name] += 1
                track_positions[tid].append({
                    "frame_idx": f_idx,
                    "x": bbox["x"],
                    "y": bbox["y"],
                    "width": bbox["width"],
                    "height": bbox["height"],
                    "bbox": bbox,
                    "confidence": det.get("confidence", 0.0),
                })

        unique_tids = list(track_positions.keys())
        tracks_data = {}
        for tid in unique_tids:
            maj_class = track_classes[tid].most_common(1)[0][0]
            tracks_data[tid] = {
                "track_id": tid,
                "vehicle_class": maj_class,
                "positions": sorted(track_positions[tid], key=lambda p: p["frame_idx"]),
            }

        analyzed_tids = [tid for tid in unique_tids if len(tracks_data[tid]["positions"]) >= self.min_track_length]

        # 3. Pairwise Spatio-Temporal Interaction Analysis
        all_interactions: List[Dict[str, Any]] = []
        hit_and_run_candidates: List[Dict[str, Any]] = []
        possible_incidents: List[Dict[str, Any]] = []

        for i in range(len(analyzed_tids)):
            for j in range(i + 1, len(analyzed_tids)):
                tid_a = analyzed_tids[i]
                tid_b = analyzed_tids[j]

                # Check interaction both ways (A suspect vs B suspect)
                res_ab = self.analyze_interaction(tracks_data[tid_a], tracks_data[tid_b], frame_width, frame_height)
                res_ba = self.analyze_interaction(tracks_data[tid_b], tracks_data[tid_a], frame_width, frame_height)

                chosen_res = res_ab if res_ab["risk_score"] >= res_ba["risk_score"] else res_ba
                all_interactions.append(chosen_res)

                if chosen_res["status"] == STATUS_HIT_AND_RUN_CANDIDATE:
                    hit_and_run_candidates.append(chosen_res)
                elif chosen_res["status"] == STATUS_POSSIBLE_INCIDENT:
                    possible_incidents.append(chosen_res)

        # 4. Sort and select strongest candidate
        all_interactions.sort(key=lambda x: x["risk_score"], reverse=True)
        hit_and_run_candidates.sort(key=lambda x: x["risk_score"], reverse=True)
        possible_incidents.sort(key=lambda x: x["risk_score"], reverse=True)

        strongest_candidate = None
        if hit_and_run_candidates:
            strongest_candidate = hit_and_run_candidates[0]
        elif possible_incidents:
            strongest_candidate = possible_incidents[0]
        elif all_interactions:
            strongest_candidate = all_interactions[0]

        # 5. ANPR Correlation on Strongest Candidate if available
        if strongest_candidate and best_frame is not None:
            suspect_tid = strongest_candidate["track_id_a"]
            # Find best frame bbox for suspect vehicle
            suspect_positions = tracks_data[suspect_tid]["positions"]
            if suspect_positions:
                target_p = suspect_positions[-1]
                suspect_bbox = target_p["bbox"]
                sx1 = max(0, int(suspect_bbox.get("x_min", 0)))
                sy1 = max(0, int(suspect_bbox.get("y_min", 0)))
                sx2 = min(frame_width, int(suspect_bbox.get("x_max", frame_width)))
                sy2 = min(frame_height, int(suspect_bbox.get("y_max", frame_height)))

                if (sx2 - sx1) > 20 and (sy2 - sy1) > 20:
                    suspect_crop = best_frame[sy1:sy2, sx1:sx2]
                    try:
                        anpr_res = self.anpr_analyzer.analyze(suspect_crop)
                        if anpr_res.get("results"):
                            best_plate = anpr_res["results"][0]
                            strongest_candidate["anpr_correlation"] = {
                                "plate_number": best_plate["plate_number"],
                                "ocr_confidence": best_plate["ocr_confidence"],
                                "plate_status": best_plate["plate_status"],
                                "is_reliable": best_plate["is_reliable"],
                            }
                    except Exception:
                        pass

        overall_status = STATUS_NO_INCIDENT
        overall_risk = 0
        overall_severity = "LOW"
        if hit_and_run_candidates:
            overall_status = STATUS_HIT_AND_RUN_CANDIDATE
            overall_risk = strongest_candidate["risk_score"]
            overall_severity = strongest_candidate["severity"]
        elif possible_incidents:
            overall_status = STATUS_POSSIBLE_INCIDENT
            overall_risk = strongest_candidate["risk_score"]
            overall_severity = strongest_candidate["severity"]
        elif strongest_candidate:
            overall_risk = strongest_candidate["risk_score"]
            overall_severity = strongest_candidate["severity"]

        elapsed_time = round(time.time() - start_time, 3)

        return {
            "success": True,
            "total_frames": total_frames,
            "processed_frames": processed_frames,
            "vehicle_detections_count": total_detections,
            "unique_tracks_count": len(unique_tids),
            "tracks_analyzed_count": len(analyzed_tids),
            "interaction_candidates_count": len(all_interactions),
            "possible_incidents_count": len(possible_incidents),
            "hit_and_run_candidates_count": len(hit_and_run_candidates),
            "hit_and_run_candidates": hit_and_run_candidates,
            "possible_incidents": possible_incidents,
            "strongest_candidate": strongest_candidate,
            "risk_score": overall_risk,
            "status": overall_status,
            "severity": overall_severity,
            "all_analyzed_interactions": all_interactions,
            "best_frame": best_frame,
            "best_frame_idx": best_frame_idx,
            "tracks_data": tracks_data,
            "processing_time_sec": elapsed_time,
        }

    def annotate_frame(
        self,
        frame: np.ndarray,
        incident_result: Dict[str, Any],
        bus_id: str = "BUS-01",
        route_name: str = "Route 17",
        is_test: bool = False,
    ) -> np.ndarray:
        """
        Annotates frame with high-contrast HUD evidence overlay:
        - Suspect vehicle bounding box & trajectory trail (orange/red gradient)
        - Impacted counterpart bounding box & trajectory trail (yellow/cyan gradient)
        - Collision proxy reticle & interaction marker
        - Top surveillance telemetry banner
        - Deterministic rule pills & risk score badge
        - ANPR correlation tag
        - Fleet telemetry footer
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        tid_a = incident_result.get("track_id_a", 1)
        tid_b = incident_result.get("track_id_b", 2)
        cls_a = incident_result.get("vehicle_class_a", "Vehicle")
        cls_b = incident_result.get("vehicle_class_b", "Vehicle")
        pos_a = incident_result.get("positions_a", [])
        pos_b = incident_result.get("positions_b", [])
        rules = incident_result.get("triggered_rules", [])
        risk_score = incident_result.get("risk_score", 10)
        severity = incident_result.get("severity", "LOW")
        status = incident_result.get("status", STATUS_NO_INCIDENT)
        metrics = incident_result.get("interaction_metrics", {})
        interact_pt = incident_result.get("interaction_point")
        anpr_corr = incident_result.get("anpr_correlation")

        # 1. Draw Trajectory Trail for Impacted Track (Track B: Cyan/Teal)
        if len(pos_b) >= 2:
            for i in range(1, len(pos_b)):
                pt1 = (int(pos_b[i - 1]["x"]), int(pos_b[i - 1]["y"]))
                pt2 = (int(pos_b[i]["x"]), int(pos_b[i]["y"]))
                alpha = float(i) / float(len(pos_b))
                color = (int(255 * (1 - alpha)), int(220 * alpha), int(200 * alpha))
                cv2.line(annotated, pt1, pt2, color, 3, cv2.LINE_AA)

        # 2. Draw Trajectory Trail for Suspect Track (Track A: Orange/Red)
        if len(pos_a) >= 2:
            for i in range(1, len(pos_a)):
                pt1 = (int(pos_a[i - 1]["x"]), int(pos_a[i - 1]["y"]))
                pt2 = (int(pos_a[i]["x"]), int(pos_a[i]["y"]))
                alpha = float(i) / float(len(pos_a))
                # BGR: Bright Orange (0, 140, 255) to Crimson (0, 0, 240)
                color = (0, int(140 * (1 - alpha * 0.7)), int(220 + 35 * alpha))
                cv2.line(annotated, pt1, pt2, color, 4, cv2.LINE_AA)

        # 3. Draw Collision Proxy Reticle & Interaction Marker
        if interact_pt:
            ix, iy = int(interact_pt["x"]), int(interact_pt["y"])
            cv2.circle(annotated, (ix, iy), 24, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.circle(annotated, (ix, iy), 12, (0, 240, 255), 2, cv2.LINE_AA)
            cv2.line(annotated, (ix - 30, iy), (ix + 30, iy), (0, 240, 255), 2, cv2.LINE_AA)
            cv2.line(annotated, (ix, iy - 30), (ix, iy + 30), (0, 240, 255), 2, cv2.LINE_AA)

            reticle_lbl = f"COLLISION PROXY ZONE ({interact_pt.get('distance_px', 0)} px)"
            cv2.putText(
                annotated,
                reticle_lbl,
                (max(10, ix - 90), max(20, iy - 32)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 240, 255),
                1,
                cv2.LINE_AA
            )

        # 4. Draw Bounding Boxes for Suspect (A) and Impacted (B) Vehicles
        def draw_vehicle_box(pos_list: List[Dict[str, Any]], tid: int, cls_name: str, is_suspect: bool):
            if not pos_list:
                return
            p = pos_list[-1]
            bbox = p.get("bbox", {})
            x1 = int(bbox.get("x_min", p["x"] - p.get("width", 50) / 2))
            y1 = int(bbox.get("y_min", p["y"] - p.get("height", 50) / 2))
            x2 = int(bbox.get("x_max", p["x"] + p.get("width", 50) / 2))
            y2 = int(bbox.get("y_max", p["y"] + p.get("height", 50) / 2))

            x1, y1 = max(0, min(w - 1, x1)), max(0, min(h - 1, y1))
            x2, y2 = max(0, min(w, x2)), max(0, min(h, y2))

            if is_suspect:
                box_color = (0, 0, 240)  # Red/Crimson
                tag_text = f"SUSPECT: #{tid} [{cls_name}]"
            else:
                box_color = (255, 200, 0)  # Cyan/Teal
                tag_text = f"IMPACTED: #{tid} [{cls_name}]"

            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)

            # Tech corner accents
            c_len = min(14, max(6, (x2 - x1) // 5))
            cv2.line(annotated, (x1, y1), (x1 + c_len, y1), (0, 255, 255), 2)
            cv2.line(annotated, (x1, y1), (x1, y1 + c_len), (0, 255, 255), 2)
            cv2.line(annotated, (x2, y1), (x2 - c_len, y1), (0, 255, 255), 2)
            cv2.line(annotated, (x2, y1), (x2, y1 + c_len), (0, 255, 255), 2)
            cv2.line(annotated, (x1, y2), (x1 + c_len, y2), (0, 255, 255), 2)
            cv2.line(annotated, (x1, y2), (x1, y2 - c_len), (0, 255, 255), 2)
            cv2.line(annotated, (x2, y2), (x2 - c_len, y2), (0, 255, 255), 2)
            cv2.line(annotated, (x2, y2), (x2, y2 - c_len), (0, 255, 255), 2)

            # Label tag
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(tag_text, font, 0.48, 1)
            ty1 = max(0, y1 - th - 6)
            ty2 = y1
            cv2.rectangle(annotated, (x1, ty1), (min(w, x1 + tw + 10), ty2), box_color, -1)
            cv2.putText(annotated, tag_text, (x1 + 4, ty2 - 3), font, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

        draw_vehicle_box(pos_b, tid_b, cls_b, is_suspect=False)
        draw_vehicle_box(pos_a, tid_a, cls_a, is_suspect=True)

        # 5. Top Surveillance Telemetry Banner
        banner_h = 56
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), (20, 24, 28), -1)
        cv2.addWeighted(overlay, 0.88, annotated, 0.12, 0, annotated)
        cv2.line(annotated, (0, banner_h), (w, banner_h), (0, 180, 255), 1)

        header_title = (
            "CONTROLLED ALGORITHM TEST EVIDENCE" if is_test
            else "URBANEYE EDGE AI — HIT & RUN INCIDENT CANDIDATE"
        )
        cv2.putText(
            annotated,
            header_title,
            (14, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (0, 220, 255),
            2,
            cv2.LINE_AA
        )

        sub_desc = "Vision-Based Incident Candidate • Kinematic & Trajectory Multi-Signal Heuristic"
        cv2.putText(
            annotated,
            sub_desc,
            (14, 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (190, 210, 210),
            1,
            cv2.LINE_AA
        )

        # Risk Score Badge on Top Right
        risk_color = (0, 0, 240) if risk_score >= 70 else (0, 160, 255)
        risk_text = f"RISK: {risk_score}/100 ({severity})"
        (rw, rh), _ = cv2.getTextSize(risk_text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
        rx1 = w - rw - 24
        cv2.rectangle(annotated, (rx1 - 8, 10), (w - 10, 44), risk_color, -1)
        cv2.putText(annotated, risk_text, (rx1 - 2, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)

        # 6. Rule Pills on Right Side
        if rules:
            pill_y = banner_h + 16
            for r in rules[:4]:
                r_label = RULE_LABELS.get(r, r.replace("_", " "))
                (pw, ph), _ = cv2.getTextSize(r_label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
                px1 = w - pw - 20
                cv2.rectangle(annotated, (px1 - 6, pill_y - 12), (w - 10, pill_y + 4), (0, 0, 180), -1)
                cv2.putText(annotated, r_label, (px1, pill_y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
                pill_y += 22

        # 7. Bottom Left HUD Telemetry Box
        hud_box_w = 340
        hud_box_h = 100
        hud_y1 = h - hud_box_h - 40
        hud_overlay = annotated.copy()
        cv2.rectangle(hud_overlay, (12, hud_y1), (12 + hud_box_w, hud_y1 + hud_box_h), (15, 20, 25), -1)
        cv2.addWeighted(hud_overlay, 0.85, annotated, 0.15, 0, annotated)
        cv2.rectangle(annotated, (12, hud_y1), (12 + hud_box_w, hud_y1 + hud_box_h), (70, 112, 126), 1)

        font_sm = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(annotated, f"TARGETS: #{tid_a} ({cls_a}) vs #{tid_b} ({cls_b})", (20, hud_y1 + 20), font_sm, 0.44, (0, 240, 255), 1, cv2.LINE_AA)
        cv2.putText(annotated, f"MIN PROXIMITY: {metrics.get('min_distance_px', 0)} px | Ratio: {metrics.get('min_proximity_ratio', 0)}", (20, hud_y1 + 40), font_sm, 0.40, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.putText(annotated, f"VELOCITY DISRUPTION: {metrics.get('max_decel_proxy', 0)} px/f² | Swerve: {metrics.get('max_swerve_deg', 0)}°", (20, hud_y1 + 60), font_sm, 0.40, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.putText(annotated, f"POST DEPARTURE SPEED: {metrics.get('post_interaction_speed_a', 0)} px/f (Fleeing Proxy)", (20, hud_y1 + 80), font_sm, 0.40, (0, 220, 255), 1, cv2.LINE_AA)

        # 8. ANPR Tag if Correlated
        if anpr_corr:
            anpr_box_w = 260
            anpr_box_h = 36
            anpr_y1 = hud_y1 - anpr_box_h - 10
            cv2.rectangle(annotated, (12, anpr_y1), (12 + anpr_box_w, anpr_y1 + anpr_box_h), (0, 120, 60), -1)
            cv2.rectangle(annotated, (12, anpr_y1), (12 + anpr_box_w, anpr_y1 + anpr_box_h), (0, 255, 120), 1)
            p_text = f"ANPR: {anpr_corr.get('plate_number', 'N/A')} ({anpr_corr.get('ocr_confidence', 0)}%)"
            cv2.putText(annotated, p_text, (20, anpr_y1 + 24), font_sm, 0.46, (255, 255, 255), 1, cv2.LINE_AA)

        # 9. Fleet Telemetry Footer
        footer_y = h - 12
        footer_text = f"FLEET EDGE AI: {bus_id} | {route_name} | Real-Time Kinematic Trajectory Engine"
        cv2.putText(annotated, footer_text, (14, footer_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 200, 200), 1, cv2.LINE_AA)

        return annotated


# Module-level singleton
_hit_and_run_analyzer_instance: Optional[HitAndRunAnalyzer] = None

def get_hit_and_run_analyzer() -> HitAndRunAnalyzer:
    global _hit_and_run_analyzer_instance
    if _hit_and_run_analyzer_instance is None:
        _hit_and_run_analyzer_instance = HitAndRunAnalyzer()
    return _hit_and_run_analyzer_instance

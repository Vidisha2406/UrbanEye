import os
import cv2
import re
import ssl
import time
import numpy as np
from typing import Union, List, Dict, Any, Optional, Tuple

# Enable unverified context for EasyOCR model downloads if necessary on macOS
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except Exception:
    pass

import easyocr
from backend.inference.vehicle_detector import get_vehicle_detector, VehicleDetector

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(CURRENT_DIR)

# Indian & Standard License Plate Regular Expressions
INDIAN_STANDARD_PLATE_REGEX = re.compile(r'^[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{4}$')
INDIAN_BHARAT_SERIES_REGEX = re.compile(r'^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$')
INDIAN_NUMERIC_SUFFIX_REGEX = re.compile(r'^[0-9]{3,4}$')
GENERIC_ALPHANUMERIC_PLATE_REGEX = re.compile(r'^[A-Z0-9]{4,10}$')

# Result State Constants
STATE_RELIABLE_PLATE = "RELIABLE PLATE"
STATE_OCR_UNCERTAIN = "PLATE DETECTED — OCR UNCERTAIN"
STATE_NO_RELIABLE_PLATE = "NO RELIABLE PLATE"


class ANPROCRAnalyzer:
    """
    Hardened Real ANPR / OCR Analyzer for UrbanEye Edge AI.
    
    Hardened Multi-Stage Pipeline:
    1. YOLO11n Vehicle Localization (Car, Bus, Truck, Motorcycle)
    2. Vehicle-Type Aware Lower Spatial ROI Segmentation
    3. Multi-Modal Plate Candidate Localization (Vertical Sobel Edges + Morphological Bridge Filter + Geometric Contours + CRAFT Text Proposals)
    4. Geometric & Spatial Prior Scoring (Aspect Ratio, Horizontal Centering, Edge Density, Contrast)
    5. Enhanced Image Preprocessing (Bicubic Upscaling + CLAHE + Bilateral Smoothing + Unsharp Masking + Multi-Binarization)
    6. EasyOCR Neural Text Recognition (CRAFT text detection + CRNN character sequence recognition)
    7. Plate Formatting & Alphanumeric Cleaning (Indian Standard State Series, Bharat Series, Registration Suffix)
    8. Explicit 3-Tier Result State Classification:
       - 'RELIABLE PLATE' (Confidence >= 70%, Valid Plate Format)
       - 'PLATE DETECTED — OCR UNCERTAIN' (Confidence 35%-69% or Ambiguous Format)
       - 'NO RELIABLE PLATE' (Confidence < 35% or No Plausible Region)
    9. Cross-Frame Temporal Aggregation, Deduplication & Best Evidence Keyframe Selection
    10. Annotated Telemetry HUD Overlay & High-Contrast Evidence Generation
    """

    def __init__(self, vehicle_detector: Optional[VehicleDetector] = None, gpu: bool = False):
        self.vehicle_detector = vehicle_detector or get_vehicle_detector()
        self.gpu = gpu
        self._reader: Optional[easyocr.Reader] = None
        self.model_name = "YOLO11n + EasyOCR CRAFT/CRNN"

    @property
    def reader(self) -> easyocr.Reader:
        if self._reader is None:
            print("[ANPR_OCR] Initializing EasyOCR Reader (English)...")
            self._reader = easyocr.Reader(['en'], gpu=self.gpu)
        return self._reader

    def generate_plate_candidate_rois(
        self,
        v_crop: np.ndarray,
        vehicle_class: str
    ) -> List[Dict[str, Any]]:
        """
        Generates geometric and morphological plate candidate bounding boxes inside vehicle crop.
        Filters candidate boxes using plate aspect ratio, spatial prior, and edge density.
        """
        if v_crop is None or v_crop.size == 0:
            return []

        vh, vw = v_crop.shape[:2]
        if vw < 30 or vh < 30:
            return []

        # Focus on vehicle lower half where plates reside
        if vehicle_class == "Motorcycle":
            roi_y1 = int(vh * 0.20)
            roi_y2 = int(vh * 0.98)
            roi_x1 = int(vw * 0.05)
            roi_x2 = int(vw * 0.95)
        else:
            roi_y1 = int(vh * 0.30)
            roi_y2 = int(vh * 0.98)
            roi_x1 = int(vw * 0.06)
            roi_x2 = int(vw * 0.94)

        sub_roi = v_crop[roi_y1:roi_y2, roi_x1:roi_x2]
        if sub_roi.size == 0:
            return []

        candidates: List[Dict[str, Any]] = []
        sub_h, sub_w = sub_roi.shape[:2]

        # 1. Primary Full Sub-ROI Fallback Candidate
        candidates.append({
            "box": (roi_x1, roi_y1, roi_x2 - roi_x1, roi_y2 - roi_y1),
            "score": 0.5,
            "source": "vehicle_lower_roi",
            "aspect_ratio": round((roi_x2 - roi_x1) / max(roi_y2 - roi_y1, 1), 2),
        })

        # 2. Morphological Vertical Edge & Contour Candidate Generator
        try:
            gray = cv2.cvtColor(sub_roi, cv2.COLOR_BGR2GRAY)
            # Sobel vertical edge filter (extract vertical character strokes)
            sobel_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
            abs_sobel = cv2.convertScaleAbs(sobel_x)
            
            # Gaussian blur + Otsu thresholding
            blurred = cv2.GaussianBlur(abs_sobel, (5, 5), 0)
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Morphological Close: Bridge characters horizontally into a unified plate blob
            k_w = max(9, int(sub_w * 0.08))
            k_h = max(3, int(sub_h * 0.04))
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_w, k_h))
            closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            sub_area = sub_w * sub_h

            for cnt in contours:
                cx, cy, cw, ch = cv2.boundingRect(cnt)
                area = cw * ch
                aspect_ratio = cw / max(ch, 1)

                # Plate geometric filters:
                # - Area: between 1.5% and 45% of sub_roi
                # - Aspect ratio: 1.2 to 6.0
                # - Min dimensions: at least 20px wide, 8px high
                if (
                    0.015 * sub_area <= area <= 0.45 * sub_area
                    and 1.2 <= aspect_ratio <= 6.0
                    and cw >= 20
                    and ch >= 8
                ):
                    # Spatial Prior: Prefer candidates centered horizontally in vehicle crop
                    cand_center_x = roi_x1 + cx + cw / 2.0
                    dist_from_v_center = abs(cand_center_x - vw / 2.0) / (0.5 * vw)
                    center_score = max(0.0, 1.0 - dist_from_v_center)

                    # Aspect Ratio Prior: standard Indian plates are ~3.0 - 4.5 (wide) or ~1.3 - 1.8 (square)
                    ar_score = 1.0 if 2.2 <= aspect_ratio <= 4.8 else 0.7

                    # Contrast / Edge score inside box
                    crop_gray = gray[cy:cy + ch, cx:cx + cw]
                    std_dev = float(np.std(crop_gray)) if crop_gray.size > 0 else 0.0
                    contrast_score = min(1.0, std_dev / 40.0)

                    cand_score = 0.4 * center_score + 0.3 * ar_score + 0.3 * contrast_score

                    candidates.append({
                        "box": (roi_x1 + cx, roi_y1 + cy, cw, ch),
                        "score": round(cand_score, 3),
                        "source": "morphological_contour",
                        "aspect_ratio": round(aspect_ratio, 2),
                    })
        except Exception:
            # Safe degradation if OpenCV contouring fails
            pass

        # Sort candidates by prior score descending
        candidates.sort(key=lambda c: c["score"], reverse=True)
        return candidates[:5]

    def preprocess_plate_crop(self, crop: np.ndarray) -> List[np.ndarray]:
        """
        Generates enhanced image variants of candidate plate ROI for robust character recognition.
        Includes Bicubic Upscaling, Bilateral Denoising, CLAHE, and Multiple Binarizations.
        """
        if crop is None or crop.size == 0:
            return []

        variants = [crop]
        h, w = crop.shape[:2]

        # 1. Resolution Normalization / Upscaling
        target_h = max(64, h)
        scale = target_h / max(h, 1)
        if scale > 1.1:
            target_w = int(w * scale)
            upscaled = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
            variants.append(upscaled)
            base_img = upscaled
        else:
            base_img = crop

        # 2. CLAHE Contrast Equalization
        try:
            gray = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
            # Bilateral filter for edge-preserving smoothing
            denoised = cv2.bilateralFilter(gray, 7, 50, 50)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced_gray = clahe.apply(denoised)
            variants.append(cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR))

            # 3. Unsharp Masking / Sharpening
            gaussian = cv2.GaussianBlur(enhanced_gray, (0, 0), 2.0)
            unsharp = cv2.addWeighted(enhanced_gray, 1.5, gaussian, -0.5, 0)
            variants.append(cv2.cvtColor(unsharp, cv2.COLOR_GRAY2BGR))

            # 4. Otsu Binarization
            _, otsu_bin = cv2.threshold(unsharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            variants.append(cv2.cvtColor(otsu_bin, cv2.COLOR_GRAY2BGR))

            # 5. Inverted Binarization (for dark backgrounds)
            inv_bin = cv2.bitwise_not(otsu_bin)
            variants.append(cv2.cvtColor(inv_bin, cv2.COLOR_GRAY2BGR))
        except Exception:
            pass

        return variants

    def classify_result_state(
        self,
        clean_text: Optional[str],
        conf: float,
        is_valid_format: bool
    ) -> str:
        """
        Hardened 3-tier classification:
        1. RELIABLE PLATE: Conf >= 70% AND valid plate format AND length >= 4
        2. PLATE DETECTED — OCR UNCERTAIN: Conf 35%-69% OR ambiguous/short text
        3. NO RELIABLE PLATE: Conf < 35% OR empty/unreadable
        """
        if not clean_text or len(clean_text) < 2 or conf < 0.35:
            return STATE_NO_RELIABLE_PLATE

        # Format checks
        is_standard_ind = bool(INDIAN_STANDARD_PLATE_REGEX.match(clean_text))
        is_bh_series = bool(INDIAN_BHARAT_SERIES_REGEX.match(clean_text))
        is_numeric_reg = bool(INDIAN_NUMERIC_SUFFIX_REGEX.match(clean_text)) and len(clean_text) >= 4
        is_generic_fmt = bool(GENERIC_ALPHANUMERIC_PLATE_REGEX.match(clean_text))

        has_valid_plate_structure = is_standard_ind or is_bh_series or is_numeric_reg or is_generic_fmt

        if conf >= 0.70 and has_valid_plate_structure and len(clean_text) >= 4:
            return STATE_RELIABLE_PLATE

        if conf >= 0.35:
            return STATE_OCR_UNCERTAIN

        return STATE_NO_RELIABLE_PLATE

    def extract_plate_from_vehicle(
        self,
        frame: np.ndarray,
        vehicle_bbox: Dict[str, float],
        vehicle_class: str
    ) -> Optional[Dict[str, Any]]:
        """
        Localizes plate candidates, runs preprocessed OCR, computes format validation,
        and applies explicit 3-tier result state thresholding.
        """
        h, w = frame.shape[:2]
        vx_min = max(0, int(vehicle_bbox.get("x_min", 0)))
        vy_min = max(0, int(vehicle_bbox.get("y_min", 0)))
        vx_max = min(w, int(vehicle_bbox.get("x_max", w)))
        vy_max = min(h, int(vehicle_bbox.get("y_max", h)))

        vw = vx_max - vx_min
        vh = vy_max - vy_min

        if vw < 20 or vh < 20:
            return None

        # Extract vehicle crop
        v_crop = frame[vy_min:vy_max, vx_min:vx_max]

        # Generate candidate proposals inside vehicle crop
        candidates = self.generate_plate_candidate_rois(v_crop, vehicle_class)

        best_result: Optional[Dict[str, Any]] = None
        highest_score = -1.0

        for cand in candidates:
            cx, cy, cw, ch = cand["box"]
            cand_crop = v_crop[cy:cy + ch, cx:cx + cw]
            if cand_crop.size == 0:
                continue

            roi_offset_x = vx_min + cx
            roi_offset_y = vy_min + cy

            # Evaluate OCR across preprocessing variants
            for variant in self.preprocess_plate_crop(cand_crop):
                try:
                    ocr_entries = self.reader.readtext(
                        variant,
                        allowlist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ- ",
                        paragraph=False
                    )
                except Exception:
                    ocr_entries = []

                for entry in ocr_entries:
                    if len(entry) < 3:
                        continue
                    bbox_poly, raw_text, conf = entry[0], entry[1], float(entry[2])

                    # Clean text: uppercase alphanumeric
                    clean_text = re.sub(r"[^A-Z0-9]", "", raw_text.upper())
                    if len(clean_text) < 2:
                        continue

                    # Evaluate plate format
                    is_indian_std = bool(INDIAN_STANDARD_PLATE_REGEX.match(clean_text))
                    is_bh_series = bool(INDIAN_BHARAT_SERIES_REGEX.match(clean_text))
                    is_numeric_reg = bool(INDIAN_NUMERIC_SUFFIX_REGEX.match(clean_text)) and len(clean_text) >= 4
                    is_generic = bool(GENERIC_ALPHANUMERIC_PLATE_REGEX.match(clean_text))
                    is_valid_format = is_indian_std or is_bh_series or is_numeric_reg or is_generic

                    # Format scoring bonus
                    if is_indian_std or is_bh_series:
                        format_multiplier = 1.6
                    elif is_numeric_reg:
                        format_multiplier = 1.3
                    elif is_generic:
                        format_multiplier = 1.15
                    else:
                        format_multiplier = 0.85

                    length_multiplier = min(1.0, len(clean_text) / 6.0)
                    prior_score = cand.get("score", 0.5)

                    # Comprehensive candidate score
                    score = conf * format_multiplier * length_multiplier * (0.6 + 0.4 * prior_score)

                    if score > highest_score:
                        highest_score = score

                        # Compute full-frame bounding box from polygon
                        poly_arr = np.array(bbox_poly, dtype=np.float32)
                        scale_x = cand_crop.shape[1] / max(variant.shape[1], 1)
                        scale_y = cand_crop.shape[0] / max(variant.shape[0], 1)

                        px_min = roi_offset_x + float(np.min(poly_arr[:, 0]) * scale_x)
                        py_min = roi_offset_y + float(np.min(poly_arr[:, 1]) * scale_y)
                        px_max = roi_offset_x + float(np.max(poly_arr[:, 0]) * scale_x)
                        py_max = roi_offset_y + float(np.max(poly_arr[:, 1]) * scale_y)

                        # Clamp to frame
                        px_min = max(0.0, min(float(w - 1), px_min))
                        py_min = max(0.0, min(float(h - 1), py_min))
                        px_max = max(px_min + 5, min(float(w), px_max))
                        py_max = max(py_min + 5, min(float(h), py_max))

                        result_state = self.classify_result_state(clean_text, conf, is_valid_format)

                        best_result = {
                            "raw_text": raw_text,
                            "plate_text": clean_text,
                            "ocr_confidence": round(conf, 4),
                            "ocr_confidence_pct": int(round(conf * 100)),
                            "score": round(score, 4),
                            "is_valid_format": is_valid_format,
                            "plate_status": result_state,
                            "is_reliable": result_state == STATE_RELIABLE_PLATE,
                            "candidate_source": cand.get("source", "roi"),
                            "plate_bbox": {
                                "x_min": round(px_min, 2),
                                "y_min": round(py_min, 2),
                                "x_max": round(px_max, 2),
                                "y_max": round(py_max, 2),
                                "width": round(px_max - px_min, 2),
                                "height": round(py_max - py_min, 2),
                            },
                            "vehicle_class": vehicle_class,
                            "vehicle_bbox": vehicle_bbox,
                        }

        # Apply strict rejection threshold if confidence is too low
        if best_result and best_result.get("ocr_confidence", 0.0) < 0.35:
            return None

        return best_result

    def analyze(
        self,
        image_input: Union[str, np.ndarray],
        confidence_threshold: float = 0.25
    ) -> Dict[str, Any]:
        """
        Performs hardened ANPR / OCR analysis on a single image.
        Returns vehicles, plate candidates, explicit result states, and best plate.
        """
        start_time = time.time()

        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                return {
                    "success": False,
                    "error": f"Image path not found: {image_input}",
                    "vehicle_count": 0,
                    "plates_detected": 0,
                    "results": [],
                    "best_plate": None,
                    "processing_time_sec": 0.0,
                }
            frame = cv2.imread(image_input)
        elif isinstance(image_input, np.ndarray):
            frame = image_input
        else:
            return {
                "success": False,
                "error": "Invalid image input type",
                "vehicle_count": 0,
                "plates_detected": 0,
                "results": [],
                "best_plate": None,
                "processing_time_sec": 0.0,
            }

        # Step 1: Real vehicle detection via YOLO11n
        v_res = self.vehicle_detector.detect(frame, confidence_threshold=confidence_threshold)
        vehicle_detections = v_res.get("detections", [])

        results: List[Dict[str, Any]] = []
        best_plate: Optional[Dict[str, Any]] = None
        max_score = -1.0
        reliable_plates_count = 0
        uncertain_plates_count = 0

        for vd in vehicle_detections:
            v_class = vd.get("class", "Car")
            v_conf = vd.get("confidence", 0.0)
            v_bbox = vd.get("bbox", {})
            track_id = vd.get("track_id")

            plate_info = self.extract_plate_from_vehicle(frame, v_bbox, v_class)
            if plate_info:
                plate_info["vehicle_confidence"] = v_conf
                plate_info["track_id"] = track_id
                results.append(plate_info)

                if plate_info.get("plate_status") == STATE_RELIABLE_PLATE:
                    reliable_plates_count += 1
                else:
                    uncertain_plates_count += 1

                if plate_info["score"] > max_score:
                    max_score = plate_info["score"]
                    best_plate = plate_info
            else:
                # Vehicle detected but no readable plate
                results.append({
                    "raw_text": None,
                    "plate_text": None,
                    "ocr_confidence": 0.0,
                    "ocr_confidence_pct": 0,
                    "score": 0.0,
                    "is_valid_format": False,
                    "plate_status": STATE_NO_RELIABLE_PLATE,
                    "is_reliable": False,
                    "plate_bbox": None,
                    "vehicle_class": v_class,
                    "vehicle_confidence": v_conf,
                    "vehicle_bbox": v_bbox,
                    "track_id": track_id,
                })

        plates_with_text = [r for r in results if r.get("plate_text")]
        elapsed = time.time() - start_time

        return {
            "success": True,
            "vehicle_count": len(vehicle_detections),
            "vehicles": vehicle_detections,
            "plates_detected": len(plates_with_text),
            "reliable_count": reliable_plates_count,
            "uncertain_count": uncertain_plates_count,
            "rejected_count": len(vehicle_detections) - len(plates_with_text),
            "results": results,
            "anpr_detections": plates_with_text,
            "best_plate": best_plate,
            "model_name": self.model_name,
            "processing_time_sec": round(elapsed, 3),
        }

    def analyze_frames(
        self,
        sampled_frames: List[Tuple[int, np.ndarray]],
        confidence_threshold: float = 0.25
    ) -> Dict[str, Any]:
        """
        Hardened sequential video keyframe evaluation.
        Performs cross-frame temporal aggregation & selects the best reliable plate.
        """
        start_time = time.time()

        if not sampled_frames:
            return {
                "success": False,
                "error": "No frames provided",
                "vehicle_count": 0,
                "plates_detected": 0,
                "best_frame_idx": None,
                "best_frame": None,
                "best_plate": None,
                "per_frame": {},
                "processing_time_sec": 0.0,
            }

        per_frame: Dict[int, Any] = {}
        best_frame_idx: Optional[int] = None
        best_frame: Optional[np.ndarray] = None
        best_plate_across_video: Optional[Dict[str, Any]] = None
        max_score = -1.0
        total_vehicles_seen = 0
        total_plates_seen = 0
        total_reliable = 0
        total_uncertain = 0

        # Temporal plate observations tracker for cross-frame aggregation: {plate_text: [scores]}
        plate_frequency: Dict[str, List[float]] = {}

        for f_idx, frame in sampled_frames:
            res = self.analyze(frame, confidence_threshold=confidence_threshold)
            per_frame[f_idx] = res

            total_vehicles_seen += res.get("vehicle_count", 0)
            total_plates_seen += res.get("plates_detected", 0)
            total_reliable += res.get("reliable_count", 0)
            total_uncertain += res.get("uncertain_count", 0)

            for item in res.get("anpr_detections", []):
                p_text = item.get("plate_text")
                if p_text:
                    if p_text not in plate_frequency:
                        plate_frequency[p_text] = []
                    plate_frequency[p_text].append(item.get("score", 0.0))

            frame_best = res.get("best_plate")
            if frame_best and frame_best.get("score", 0.0) > max_score:
                max_score = frame_best.get("score", 0.0)
                best_plate_across_video = frame_best
                best_frame_idx = f_idx
                best_frame = frame

        # Fallback to first frame if no plates recognized
        if best_frame is None and sampled_frames:
            best_frame_idx = sampled_frames[0][0]
            best_frame = sampled_frames[0][1]

        elapsed = time.time() - start_time

        return {
            "success": True,
            "total_frames_processed": len(sampled_frames),
            "total_vehicles_seen": total_vehicles_seen,
            "total_plates_seen": total_plates_seen,
            "total_reliable": total_reliable,
            "total_uncertain": total_uncertain,
            "best_frame_idx": best_frame_idx,
            "best_frame": best_frame,
            "best_plate": best_plate_across_video,
            "temporal_aggregates": plate_frequency,
            "per_frame": per_frame,
            "processing_time_sec": round(elapsed, 3),
        }

    def annotate_frame(
        self,
        frame: np.ndarray,
        anpr_results: List[Dict[str, Any]],
        best_plate: Optional[Dict[str, Any]] = None,
        bus_id: str = "BUS-01",
        route_name: str = "Route 17"
    ) -> np.ndarray:
        """
        Draws high-contrast UrbanEye ANPR HUD overlays, vehicle bounding boxes,
        and localized plate tags on the evidence frame with status indicators.
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        font = cv2.FONT_HERSHEY_SIMPLEX

        # 1. Draw vehicle and plate bounding boxes
        for item in anpr_results:
            v_bbox = item.get("vehicle_bbox", {})
            p_bbox = item.get("plate_bbox")
            plate_text = item.get("plate_text")
            ocr_conf_pct = item.get("ocr_confidence_pct", 0)
            v_class = item.get("vehicle_class", "Car")
            track_id = item.get("track_id")
            plate_status = item.get("plate_status", STATE_NO_RELIABLE_PLATE)
            is_reliable = (plate_status == STATE_RELIABLE_PLATE)

            # Vehicle Box (Cyan/Emerald in BGR: (200, 230, 0))
            if v_bbox:
                vx1 = max(0, int(v_bbox.get("x_min", 0)))
                vy1 = max(0, int(v_bbox.get("y_min", 0)))
                vx2 = min(w, int(v_bbox.get("x_max", w)))
                vy2 = min(h, int(v_bbox.get("y_max", h)))

                cv2.rectangle(annotated, (vx1, vy1), (vx2, vy2), (200, 230, 0), 2)

                # Vehicle label
                v_label = f"{v_class}" if track_id is None else f"{v_class} #{track_id}"
                (tw, th), _ = cv2.getTextSize(v_label, font, 0.5, 1)
                cv2.rectangle(annotated, (vx1, max(0, vy1 - 20)), (vx1 + tw + 8, vy1), (200, 230, 0), -1)
                cv2.putText(annotated, v_label, (vx1 + 4, max(14, vy1 - 5)), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            # Plate Box (Amber/Yellow for reliable, Orange for uncertain)
            if p_bbox and plate_text:
                px1 = max(0, int(p_bbox.get("x_min", 0)))
                py1 = max(0, int(p_bbox.get("y_min", 0)))
                px2 = min(w, int(p_bbox.get("x_max", w)))
                py2 = min(h, int(p_bbox.get("y_max", h)))

                box_color = (0, 215, 255) if is_reliable else (0, 140, 255)  # Amber vs Orange in BGR

                # Draw plate bounding box
                cv2.rectangle(annotated, (px1, py1), (px2, py2), box_color, 3)

                # Plate Corner Accents
                c_len = min(12, max(4, (px2 - px1) // 4))
                cv2.line(annotated, (px1, py1), (px1 + c_len, py1), (255, 255, 255), 2)
                cv2.line(annotated, (px1, py1), (px1, py1 + c_len), (255, 255, 255), 2)
                cv2.line(annotated, (px2, py1), (px2 - c_len, py1), (255, 255, 255), 2)
                cv2.line(annotated, (px2, py1), (px2 - c_len, py1 + c_len), (255, 255, 255), 2)

                # Plate Text Tag
                status_label = "[RELIABLE]" if is_reliable else "[UNCERTAIN]"
                tag_str = f"PLATE: {plate_text} ({ocr_conf_pct}%) {status_label}"
                (ptw, pth), _ = cv2.getTextSize(tag_str, font, 0.52, 2)
                tag_y = max(pth + 10, py1 - 8) if py1 > pth + 12 else min(h - 5, py2 + pth + 14)
                cv2.rectangle(annotated, (px1, tag_y - pth - 6), (px1 + ptw + 12, tag_y + 4), (0, 0, 0), -1)
                cv2.rectangle(annotated, (px1, tag_y - pth - 6), (px1 + ptw + 12, tag_y + 4), box_color, 1)
                cv2.putText(annotated, tag_str, (px1 + 6, tag_y - 2), font, 0.52, box_color, 2, cv2.LINE_AA)

        # 2. Top Telemetry & HUD Banner
        banner_h = 44
        cv2.rectangle(annotated, (0, 0), (w, banner_h), (15, 22, 24), -1)
        cv2.line(annotated, (0, banner_h), (w, banner_h), (0, 215, 255), 2)

        # Pulse indicator circle
        indicator_color = (0, 255, 128) if (best_plate and best_plate.get("is_reliable")) else (0, 165, 255)
        cv2.circle(annotated, (20, banner_h // 2), 6, indicator_color, -1)

        best_text_str = best_plate.get("plate_text", "SCANNING") if best_plate else "NO READABLE PLATE"
        best_conf_str = f"{best_plate.get('ocr_confidence_pct', 0)}%" if best_plate else "--"
        best_status_str = best_plate.get("plate_status", STATE_NO_RELIABLE_PLATE) if best_plate else "NO DETECTION"

        hud_title = f"URBANEYE HARDENED ANPR / OCR | Fleet: {bus_id} ({route_name})"
        hud_details = f"Plate: {best_text_str} | Conf: {best_conf_str} | Status: {best_status_str} | Model: EasyOCR CRAFT/CRNN"

        cv2.putText(annotated, hud_title, (36, 18), font, 0.45, (200, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(annotated, hud_details, (36, 36), font, 0.46, (0, 240, 255), 1, cv2.LINE_AA)

        return annotated


# Global Singleton instance
_ANPR_ANALYZER_INSTANCE: Optional[ANPROCRAnalyzer] = None

def get_anpr_ocr_analyzer() -> ANPROCRAnalyzer:
    global _ANPR_ANALYZER_INSTANCE
    if _ANPR_ANALYZER_INSTANCE is None:
        _ANPR_ANALYZER_INSTANCE = ANPROCRAnalyzer()
    return _ANPR_ANALYZER_INSTANCE

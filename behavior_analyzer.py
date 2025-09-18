import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Tuple

import cv2
import numpy as np

# MediaPipe is imported in the runner to build a single graph; this module
# focuses on metrics computed from landmarks and simple per-frame helpers.


@dataclass
class BehaviorMetrics:
    posture: str
    head_roll_deg: float
    head_yaw_deg: float
    head_pitch_deg: float
    eyes_closed_ratio: float
    mouth_open_ratio: float
    fidget_score: float
    hand_to_face: bool
    holding_object: bool
    object_kind: str
    object_confidence: float


class BehaviorAnalyzer:
    """Compute human behavior metrics from MediaPipe Holistic landmarks.

    This class is stateless across frames except for a small motion history
    buffer used to estimate fidgeting.
    """

    def __init__(self, motion_history_seconds: float = 2.0, fps_assumption: float = 30.0) -> None:
        self.motion_buffer_size: int = max(5, int(motion_history_seconds * fps_assumption))
        self.motion_history: Deque[float] = deque(maxlen=self.motion_buffer_size)
        self._previous_pose_xy: Optional[np.ndarray] = None
        self._previous_time: Optional[float] = None

    # ----------------------------- Core API ---------------------------------
    def compute_metrics(
        self,
        frame_bgr: np.ndarray,
        pose_landmarks: Optional[object],
        face_landmarks: Optional[object],
        left_hand_landmarks: Optional[object],
        right_hand_landmarks: Optional[object],
    ) -> BehaviorMetrics:
        h, w = frame_bgr.shape[:2]

        # Posture from shoulders/hips/nose
        posture_label = self._estimate_posture(pose_landmarks, frame_size=(w, h))

        # Head pose proxies from face mesh
        head_roll, head_yaw, head_pitch, eyes_closed_ratio, mouth_open_ratio = self._estimate_head_and_face_states(
            face_landmarks, frame_size=(w, h)
        )

        # Fidgeting from pose landmark motion
        fidget_score = self._estimate_fidget(pose_landmarks, frame_size=(w, h))

        # Hand-to-face proximity
        hand_to_face_flag = self._estimate_hand_to_face(
            face_landmarks, left_hand_landmarks, right_hand_landmarks, frame_size=(w, h)
        )

        # Object in hand detection (phone/paper heuristic)
        obj_holding, obj_kind, obj_conf = self._detect_object_near_hands(
            frame_bgr, left_hand_landmarks, right_hand_landmarks, frame_size=(w, h)
        )

        return BehaviorMetrics(
            posture=posture_label,
            head_roll_deg=head_roll,
            head_yaw_deg=head_yaw,
            head_pitch_deg=head_pitch,
            eyes_closed_ratio=eyes_closed_ratio,
            mouth_open_ratio=mouth_open_ratio,
            fidget_score=fidget_score,
            hand_to_face=hand_to_face_flag,
            holding_object=obj_holding,
            object_kind=obj_kind,
            object_confidence=obj_conf,
        )

    # --------------------------- Helper methods -----------------------------
    @staticmethod
    def _landmarks_to_xy_array(landmarks: object, frame_size: Tuple[int, int]) -> Optional[np.ndarray]:
        if landmarks is None or getattr(landmarks, "landmark", None) is None:
            return None
        w, h = frame_size
        points = [(lm.x * w, lm.y * h) for lm in landmarks.landmark]
        return np.asarray(points, dtype=np.float32)

    @staticmethod
    def _safe_distance(p1: np.ndarray, p2: np.ndarray) -> float:
        return float(np.linalg.norm(p1 - p2))

    def _estimate_posture(self, pose_landmarks: Optional[object], frame_size: Tuple[int, int]) -> str:
        xy = self._landmarks_to_xy_array(pose_landmarks, frame_size)
        if xy is None or len(xy) < 33:
            return "unknown"
        # MediaPipe Pose indices
        LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
        LEFT_HIP, RIGHT_HIP = 23, 24
        NOSE = 0

        shoulder_mid = (xy[LEFT_SHOULDER] + xy[RIGHT_SHOULDER]) / 2.0
        hip_mid = (xy[LEFT_HIP] + xy[RIGHT_HIP]) / 2.0
        nose = xy[NOSE]

        spine_vec = shoulder_mid - hip_mid
        spine_len = np.linalg.norm(spine_vec)
        if spine_len < 1e-3:
            return "unknown"

        # Simple slouch proxy: nose projected horizontally ahead of shoulders, and short spine length
        head_forward = nose[0] - shoulder_mid[0]
        slouch_score = abs(head_forward) / max(1.0, spine_len)
        uprightness = spine_vec[1] / max(1.0, spine_len)  # positive if shoulders below hips in image

        if slouch_score > 0.9 or uprightness < -0.2:
            return "slouching"
        if slouch_score > 0.6:
            return "leaning"
        return "upright"

    def _estimate_head_and_face_states(
        self, face_landmarks: Optional[object], frame_size: Tuple[int, int]
    ) -> Tuple[float, float, float, float, float]:
        xy = self._landmarks_to_xy_array(face_landmarks, frame_size)
        if xy is None or len(xy) < 468:
            return 0.0, 0.0, 0.0, 0.0, 0.0

        # Common FaceMesh indices
        LEFT_EYE_OUTER, LEFT_EYE_INNER = 33, 133
        LEFT_EYE_TOP, LEFT_EYE_BOTTOM = 159, 145
        RIGHT_EYE_OUTER, RIGHT_EYE_INNER = 263, 362
        RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM = 386, 374

        MOUTH_LEFT, MOUTH_RIGHT = 61, 291
        MOUTH_TOP, MOUTH_BOTTOM = 13, 14

        NOSE_TIP = 1
        MID_EYES_LEFT, MID_EYES_RIGHT = 33, 263

        # Head roll from eye line slope
        left_eye_center = (xy[LEFT_EYE_OUTER] + xy[LEFT_EYE_INNER]) / 2.0
        right_eye_center = (xy[RIGHT_EYE_OUTER] + xy[RIGHT_EYE_INNER]) / 2.0
        eye_vec = right_eye_center - left_eye_center
        head_roll = math.degrees(math.atan2(eye_vec[1], eye_vec[0]))

        # Yaw proxy: nose x offset from mid-eye center normalized by inter-ocular distance
        mid_eye = (left_eye_center + right_eye_center) / 2.0
        inter_ocular = max(1.0, np.linalg.norm(eye_vec))
        yaw_norm = (xy[NOSE_TIP][0] - mid_eye[0]) / inter_ocular
        head_yaw = float(np.clip(yaw_norm, -1.0, 1.0)) * 35.0  # approx degrees

        # Pitch proxy: vertical distance between eyes and mouth vs inter-ocular
        mouth_center = (xy[MOUTH_LEFT] + xy[MOUTH_RIGHT]) / 2.0
        eye_mouth_v = (mouth_center[1] - mid_eye[1]) / inter_ocular
        head_pitch = float(np.clip(-(eye_mouth_v - 0.6), -1.0, 1.0)) * 25.0

        # Eye Aspect Ratio (simplified) per eye then averaged
        left_eye_open = self._vertical_over_horizontal_ratio(
            xy[LEFT_EYE_TOP], xy[LEFT_EYE_BOTTOM], xy[LEFT_EYE_OUTER], xy[LEFT_EYE_INNER]
        )
        right_eye_open = self._vertical_over_horizontal_ratio(
            xy[RIGHT_EYE_TOP], xy[RIGHT_EYE_BOTTOM], xy[RIGHT_EYE_OUTER], xy[RIGHT_EYE_INNER]
        )
        eyes_open_ratio = float((left_eye_open + right_eye_open) / 2.0)
        # Convert to closed ratio (1 means closed) by inverting within a plausible range
        eyes_closed_ratio = float(np.clip(1.0 - np.clip((eyes_open_ratio - 0.12) / 0.18, 0.0, 1.0), 0.0, 1.0))

        # Mouth Aspect Ratio
        mouth_open_ratio = self._vertical_over_horizontal_ratio(
            xy[MOUTH_TOP], xy[MOUTH_BOTTOM], xy[MOUTH_LEFT], xy[MOUTH_RIGHT]
        )

        return float(head_roll), float(head_yaw), float(head_pitch), eyes_closed_ratio, float(mouth_open_ratio)

    @staticmethod
    def _vertical_over_horizontal_ratio(top: np.ndarray, bottom: np.ndarray, left: np.ndarray, right: np.ndarray) -> float:
        vertical = np.linalg.norm(top - bottom)
        horizontal = max(1e-3, np.linalg.norm(left - right))
        return float(vertical / horizontal)

    def _estimate_fidget(self, pose_landmarks: Optional[object], frame_size: Tuple[int, int]) -> float:
        xy = self._landmarks_to_xy_array(pose_landmarks, frame_size)
        if xy is None or len(xy) < 33:
            # decay history if no detection
            if len(self.motion_history) > 0:
                self.motion_history.append(self.motion_history[-1] * 0.9)
            return float(np.mean(self.motion_history)) if len(self.motion_history) else 0.0

        # Use a subset of stable landmarks (shoulders, hips, nose)
        idxs = [0, 11, 12, 23, 24]
        subset = xy[idxs]
        now = time.time()

        if self._previous_pose_xy is None or self._previous_pose_xy.shape != subset.shape:
            self._previous_pose_xy = subset.copy()
            self._previous_time = now
            self.motion_history.append(0.0)
            return 0.0

        dt = max(1e-3, (now - (self._previous_time or now)))
        displacement = np.linalg.norm(subset - self._previous_pose_xy, axis=1).mean()
        speed = displacement / dt

        # Normalize by torso size for scale invariance
        torso = np.linalg.norm((xy[11] + xy[12]) / 2.0 - (xy[23] + xy[24]) / 2.0)
        norm_speed = float(speed / max(1.0, torso))
        self.motion_history.append(norm_speed)

        self._previous_pose_xy = subset.copy()
        self._previous_time = now

        # Return average over history
        return float(np.mean(self.motion_history))

    def _estimate_hand_to_face(
        self,
        face_landmarks: Optional[object],
        left_hand_landmarks: Optional[object],
        right_hand_landmarks: Optional[object],
        frame_size: Tuple[int, int],
    ) -> bool:
        face_xy = self._landmarks_to_xy_array(face_landmarks, frame_size)
        if face_xy is None or len(face_xy) < 468:
            return False
        face_center = face_xy.mean(axis=0)
        # Estimate face scale using mouth width
        mouth_w = np.linalg.norm(face_xy[61] - face_xy[291])
        threshold = max(20.0, mouth_w * 0.6)

        min_dist = float("inf")
        for hand in (left_hand_landmarks, right_hand_landmarks):
            hand_xy = self._landmarks_to_xy_array(hand, frame_size)
            if hand_xy is None:
                continue
            for pt in hand_xy:
                min_dist = min(min_dist, self._safe_distance(pt, face_center))

        return bool(min_dist < threshold)

    # ----------------------- Object detection heuristics ----------------------
    def _detect_object_near_hands(
        self,
        frame_bgr: np.ndarray,
        left_hand_landmarks: Optional[object],
        right_hand_landmarks: Optional[object],
        frame_size: Tuple[int, int],
    ) -> Tuple[bool, str, float]:
        """Heuristic detection of rectangular objects (phone/paper) near hands.

        Returns (holding_object, kind, confidence).
        kind in {"phone", "paper", "unknown"}.
        """
        h, w = frame_bgr.shape[:2]
        best_kind = "unknown"
        best_conf = 0.0
        holding = False

        for hand in (left_hand_landmarks, right_hand_landmarks):
            hand_xy = self._landmarks_to_xy_array(hand, frame_size)
            if hand_xy is None:
                continue
            x_min = max(0, int(np.min(hand_xy[:, 0]) - 20))
            y_min = max(0, int(np.min(hand_xy[:, 1]) - 20))
            x_max = min(w - 1, int(np.max(hand_xy[:, 0]) + 20))
            y_max = min(h - 1, int(np.max(hand_xy[:, 1]) + 20))
            if x_max - x_min < 20 or y_max - y_min < 20:
                continue

            roi = frame_bgr[y_min:y_max, x_min:x_max]
            if roi.size == 0:
                continue
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blur, 30, 120)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            area_roi = float(roi.shape[0] * roi.shape[1])
            brightness = float(np.mean(gray)) / 255.0
            edge_density = float(np.count_nonzero(edges)) / max(1.0, area_roi)

            for cnt in contours:
                area = float(cv2.contourArea(cnt))
                if area < area_roi * 0.015:
                    continue
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                rect_like = len(approx) in (4, 5, 6)
                rect = cv2.boundingRect(cnt)
                _, _, ww, hh = rect
                if ww <= 0 or hh <= 0:
                    continue
                aspect = float(max(ww, hh) / max(1.0, min(ww, hh)))
                fill_ratio = float(area / (ww * hh))

                # Classify
                conf_phone = 0.0
                conf_paper = 0.0
                if rect_like and fill_ratio > 0.5:
                    # Phone: darker OR medium brightness, elongated rectangle, decent edge density
                    if aspect >= 1.2 and aspect <= 4.0 and brightness < 0.7:
                        conf_phone = 0.5
                        conf_phone += 0.2 * min(1.0, (aspect - 1.2) / 2.8)
                        conf_phone += 0.2 * (1.0 - min(1.0, (brightness - 0.2) / 0.5))
                        conf_phone += 0.1 * min(1.0, edge_density / 0.15)
                    # Paper: bright rectangle covering larger portion of ROI
                    coverage = area / area_roi
                    if brightness > 0.6 and coverage > 0.12 and aspect <= 4.0:
                        conf_paper = 0.55
                        conf_paper += 0.25 * min(1.0, (brightness - 0.6) / 0.3)
                        conf_paper += 0.2 * min(1.0, coverage / 0.35)

                # Choose best for this contour
                if conf_phone > best_conf:
                    best_conf = conf_phone
                    best_kind = "phone"
                    holding = True
                if conf_paper > best_conf:
                    best_conf = conf_paper
                    best_kind = "paper"
                    holding = True

        if best_conf < 0.45:
            return False, "unknown", float(best_conf)
        return holding, best_kind, float(np.clip(best_conf, 0.0, 1.0))


def draw_overlay(frame_bgr: np.ndarray, metrics: BehaviorMetrics) -> None:
    """Draw a compact overlay with behavior metrics on the frame in-place."""
    h, w = frame_bgr.shape[:2]
    pad = 12
    x0, y0 = pad, pad
    panel_w, panel_h = 280, 180

    # Panel background
    cv2.rectangle(frame_bgr, (x0, y0), (x0 + panel_w, y0 + panel_h), (0, 0, 0), thickness=-1)
    cv2.addWeighted(frame_bgr[y0 : y0 + panel_h, x0 : x0 + panel_w], 0.6, frame_bgr[y0 : y0 + panel_h, x0 : x0 + panel_w], 0.4, 0, frame_bgr[y0 : y0 + panel_h, x0 : x0 + panel_w])

    def put(text: str, y: int, color: Tuple[int, int, int] = (255, 255, 255)) -> None:
        cv2.putText(frame_bgr, text, (x0 + 8, y0 + y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    put(f"Posture: {metrics.posture}", 20, (200, 220, 255))
    put(f"Head roll: {metrics.head_roll_deg:+.1f}°", 45)
    put(f"Head yaw: {metrics.head_yaw_deg:+.1f}°", 65)
    put(f"Head pitch: {metrics.head_pitch_deg:+.1f}°", 85)

    eye_state = "closed" if metrics.eyes_closed_ratio > 0.6 else ("blinking" if metrics.eyes_closed_ratio > 0.3 else "open")
    put(f"Eyes: {eye_state} ({metrics.eyes_closed_ratio:.2f})", 110)

    mouth_state = "open" if metrics.mouth_open_ratio > 0.5 else ("speaking?" if metrics.mouth_open_ratio > 0.35 else "closed")
    put(f"Mouth: {mouth_state} ({metrics.mouth_open_ratio:.2f})", 130)

    fidget_state = "high" if metrics.fidget_score > 0.012 else ("moderate" if metrics.fidget_score > 0.006 else "low")
    put(f"Fidgeting: {fidget_state} ({metrics.fidget_score:.3f})", 150)

    put(f"Hand-to-face: {'yes' if metrics.hand_to_face else 'no'}", 170, (120, 255, 120) if metrics.hand_to_face else (180, 180, 180))

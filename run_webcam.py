import sys
import time
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

from behavior_analyzer import BehaviorAnalyzer, BehaviorMetrics, draw_overlay


def main(camera_index: int = 0) -> int:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Cannot open camera index {camera_index}")
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    mp_holistic = mp.solutions.holistic
    drawing = mp.solutions.drawing_utils
    drawing_styles = mp.solutions.drawing_styles

    analyzer = BehaviorAnalyzer()

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        refine_face_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:
        last_fps_time = time.time()
        frames = 0

        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                print("Warning: camera frame grab failed")
                break

            frame_bgr = cv2.flip(frame_bgr, 1)  # mirror for user-friendly view
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            results = holistic.process(frame_rgb)

            metrics: BehaviorMetrics = analyzer.compute_metrics(
                frame_bgr,
                results.pose_landmarks,
                results.face_landmarks,
                results.left_hand_landmarks,
                results.right_hand_landmarks,
            )

            # Draw landmarks (lightweight style)
            if results.pose_landmarks is not None:
                drawing.draw_landmarks(
                    frame_bgr,
                    results.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=drawing.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=2),
                    connection_drawing_spec=drawing.DrawingSpec(color=(0, 120, 120), thickness=1),
                )
            if results.face_landmarks is not None:
                drawing.draw_landmarks(
                    frame_bgr,
                    results.face_landmarks,
                    mp_holistic.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=drawing_styles.get_default_face_mesh_tesselation_style(),
                )
            if results.left_hand_landmarks is not None:
                drawing.draw_landmarks(
                    frame_bgr,
                    results.left_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=drawing.DrawingSpec(color=(255, 200, 0), thickness=1, circle_radius=2),
                    connection_drawing_spec=drawing.DrawingSpec(color=(160, 140, 60), thickness=1),
                )
            if results.right_hand_landmarks is not None:
                drawing.draw_landmarks(
                    frame_bgr,
                    results.right_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=drawing.DrawingSpec(color=(0, 200, 255), thickness=1, circle_radius=2),
                    connection_drawing_spec=drawing.DrawingSpec(color=(60, 140, 160), thickness=1),
                )

            draw_overlay(frame_bgr, metrics)

            frames += 1
            if frames >= 10:
                now = time.time()
                fps = frames / (now - last_fps_time)
                last_fps_time = now
                frames = 0
                cv2.putText(frame_bgr, f"FPS: {fps:.1f}", (10, frame_bgr.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow("Behavior Analyzer", frame_bgr)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    cam_idx = 0
    if len(sys.argv) > 1:
        try:
            cam_idx = int(sys.argv[1])
        except ValueError:
            print("Usage: python run_webcam.py [camera_index]")
            sys.exit(2)
    sys.exit(main(cam_idx))

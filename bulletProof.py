import os, sys, cv2, time, math
import mediapipe as mp
from collections import deque, Counter

# --- If you're on Wayland and using OpenCV's Qt backend, uncomment this:
# os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

mp_pose  = mp.solutions.pose
mp_draw  = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles

def draw_and_label(frame_bgr, results, label, fps):
    """Draw landmarks (expects RGB for MediaPipe drawing, then convert back to BGR for imshow)."""
    if results.pose_landmarks:
        # Convert to RGB for drawing utils
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_draw.draw_landmarks(
            rgb, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_style.get_default_pose_landmarks_style()
        )
        # Back to BGR for imshow
        frame_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    # HUD
    cv2.putText(frame_bgr, f"FPS {fps:4.1f}", (8,24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(frame_bgr, f"Pose: {label or '(none)'}", (8,52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50,220,50), 2)
    return frame_bgr

def main():
    # If another process is using the camera, close it first!
    # Try explicit backend on Linux:
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("Camera failed to open via V4L2, trying CAP_ANY…")
        cap = cv2.VideoCapture(0, cv2.CAP_ANY)
    if not cap.isOpened():
        print("ERROR: could not open camera 0. Is another app using it?")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Nice window flags for Linux
    cv2.namedWindow("MediaPipe Pose – debug", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow("MediaPipe Pose – debug", 960, 540)

    pose = mp_pose.Pose(model_complexity=0)

    # minimal labeler (you can plug your rules back here)
    def label_pose(res):
        return "tracking" if res.pose_landmarks else ""

    prev = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            print("WARN: cap.read() returned False — camera feed not available.")
            break

        # Mirror for user view
        frame = cv2.flip(frame, 1)

        # Sanity print once
        if int(time.time()) % 5 == 0:
            h, w = frame.shape[:2]
            # print every 5s so it doesn't spam
            print(f"Frame shape: {w}x{h}")

        # MediaPipe expects RGB input for inference
        res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        now = time.time()
        fps = 1.0 / max(1e-6, (now - prev))
        prev = now

        disp = draw_and_label(frame, res, label_pose(res), fps)
        cv2.imshow("MediaPipe Pose – debug", disp)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


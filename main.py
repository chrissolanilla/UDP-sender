import socket, json, time, cv2
import mediapipe as mp
from collections import deque, Counter
from gestures import detect_gestures

ADDR = ("127.0.0.1", 54545)
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(model_complexity=0, enable_segmentation=False)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# --- Stability state ---
WINDOW = 10                 # frames to vote over
COOLDOWN = 0.25             # seconds min between changes (debounce)
hist = deque(maxlen=WINDOW)
last_stable = None
last_change = 0.0

def to_packet(results, fps, gestures_raw, gesture, changed, tracking):
    lm = []
    if results.pose_landmarks:
        for i, p in enumerate(results.pose_landmarks.landmark):
            lm.append({"id": i,
                       "x": round(float(p.x),3),
                       "y": round(float(p.y),3),
                       "v": round(float(p.visibility),3)})
    return {
        "t": time.time(),
        "fps": round(fps,2),
        "tracking": tracking,         # True if landmarks present
        "landmarks": lm,              # raw landmarks if you need them
        "gestures_raw": gestures_raw, # noisy per-frame list
        "gesture": gesture,           # single, stable label or ""
        "changed": changed            # True only on edge (state change)
    }

prev = time.time()
while True:
    ok, frame = cap.read()
    if not ok: break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    # Per-frame detections
    gestures_raw = []
    tracking = results.pose_landmarks is not None
    if tracking:
        gestures_raw = detect_gestures(results.pose_landmarks.landmark)

        # optional: prioritize one if multiple; or keep first
        if any("Scratching" in g for g in gestures_raw):
            gestures_raw = [g for g in gestures_raw if "Scratching" in g]
        elif len(gestures_raw) > 1:
            gestures_raw = [gestures_raw[0]]
    else:
        gestures_raw = []

    # Append a single label (or "") for the voting window
    label = gestures_raw[0] if gestures_raw else ""
    hist.append(label)

    # Majority vote when window full
    gesture = last_stable
    changed = False
    if len(hist) == hist.maxlen:
        # pick the most common non-empty label (or empty if none)
        nonempty = [g for g in hist if g]
        voted = Counter(nonempty).most_common(1)[0][0] if nonempty else ""
        now = time.time()
        if voted != last_stable and (now - last_change) >= COOLDOWN:
            last_stable = voted
            last_change = now
            gesture = voted
            changed = True

    now = time.time()
    fps = 1.0 / max(1e-6, (now - prev)); prev = now

    pkt = to_packet(results, fps, gestures_raw, gesture, changed, tracking)
    buf = json.dumps(pkt, separators=(",",":"), allow_nan=False).encode("utf-8")
    sock.sendto(buf, ADDR)

# import socket, json, time, cv2
# import mediapipe as mp
#
# from gestures import detect_gestures
#
# ADDR = ("127.0.0.1", 54545)  # change IP to your Godot box if needed
# sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose(model_complexity=0, enable_segmentation=False)
# cap = cv2.VideoCapture(0)    # consider 640x480 for speed
#
# def to_packet(results, fps, gestures):
#     lm = []
#     if results.pose_landmarks:
#         for i, p in enumerate(results.pose_landmarks.landmark):
#             lm.append({"id": i, "x": p.x, "y": p.y, "v": p.visibility})
#     return {
#         "t": time.time(),
#         "fps": fps,
#         "landmarks": lm,
#         "gestures": gestures
#     }
#
# prev = time.time()
# while True:
#     ok, frame = cap.read()
#     if not ok: break
#     frame = cv2.flip(frame, 1)  # mirrored user view
#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = pose.process(rgb)
#
#     # your gesture function from the prompt (works on results.pose_landmarks.landmark)
#     def get_landmark(landmarks, name):
#         return landmarks[mp_pose.PoseLandmark[name].value]
#     def distance(a, b):
#         return ((a.x - b.x)**2 + (a.y - b.y)**2) ** 0.5
#     gestures = []
#     if results.pose_landmarks:
#         # lms = results.pose_landmarks.landmark
#         gestures = detect_gestures(results.pose_landmarks.landmark)
#
#     now = time.time()
#     fps = 1.0 / max(1e-6, (now - prev))
#     prev = now
#
#     packet = to_packet(results, fps, gestures)
#     buf = json.dumps(packet, separators=(",",":")).encode("utf-8")
#     # Keep it tiny; drop if > ~1300 bytes or thin landmarks (e.g., every 2nd frame)
#     sock.sendto(buf, ADDR)
#

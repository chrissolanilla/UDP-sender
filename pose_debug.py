import os, sys, cv2, time, math
import mediapipe as mp
from collections import deque, Counter

# If Wayland/Qt gives blank windows, uncomment:
# os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

mp_pose  = mp.solutions.pose
mp_draw  = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles
M = mp_pose.PoseLandmark

# ---------- helpers ----------
def pt(lms, name): return lms[M[name].value]

def vis_ok(p, th=0.6):
    return (getattr(p, "visibility", 0.0) or 0.0) >= th and -0.05 <= p.x <= 1.05 and -0.05 <= p.y <= 1.05

def dist(a, b): return math.hypot(a.x - b.x, a.y - b.y)

def angle(a, b, c):
    abx, aby = a.x - b.x, a.y - b.y
    cbx, cby = c.x - b.x, c.y - b.y
    da = math.hypot(abx, aby); dc = math.hypot(cbx, cby)
    if da*dc == 0: return 180.0
    cosv = max(-1.0, min(1.0, (abx*cbx + aby*cby)/(da*dc)))
    return math.degrees(math.acos(cosv))

def torso_scale(lms):
    """Reference length to make thresholds resolution/body-size invariant."""
    ls, rs = pt(lms,"LEFT_SHOULDER"), pt(lms,"RIGHT_SHOULDER")
    lh, rh = pt(lms,"LEFT_HIP"), pt(lms,"RIGHT_HIP")
    if not all(map(vis_ok, (ls,rs,lh,rh))): return 0.35
    sh = dist(ls, rs)
    th = 0.5*(dist(ls, lh) + dist(rs, rh))
    return max(0.25, min(0.7, 0.6*sh + 0.4*th))

def above(p, q, margin=0.0):  # y goes down; smaller y is higher
    return p.y < q.y - margin

def near(a, b, s, k):  # distance < k * scale
    return dist(a,b) <= (k * s)

def horiz_aligned(a, b, s, eps=0.12):
    return abs(a.y - b.y) <= eps * s

def between_vertical(p, top, bottom, s, top_eps=0.12, bot_eps=0.10):
    # Ensures p is between top and bottom (inclusive, with margins).
    # y grows down; "top" is visually higher (smaller y).
    top_y = min(top.y, bottom.y)   # the visually higher one
    bot_y = max(top.y, bottom.y)   # the visually lower one
    return (p.y >= top_y - top_eps * s) and (p.y <= bot_y + bot_eps * s)

# ---------- rules for your 6 poses ----------
def classify_pose(lms):
    """
    Returns one of:
    'Tough Guy Pose', 'Muscle Man Pose', 'What? Pose',
    'Point Up Pose (L/R)', 'Samurai Pose', 'Mantis Pose', or ''.
    """
    s = torso_scale(lms)

    lw, rw = pt(lms,"LEFT_WRIST"), pt(lms,"RIGHT_WRIST")
    le, re = pt(lms,"LEFT_ELBOW"), pt(lms,"RIGHT_ELBOW")
    ls, rs = pt(lms,"LEFT_SHOULDER"), pt(lms,"RIGHT_SHOULDER")
    lh, rh = pt(lms,"LEFT_HIP"), pt(lms,"RIGHT_HIP")
    lk, rk = pt(lms,"LEFT_KNEE"), pt(lms,"RIGHT_KNEE")
    leye = pt(lms, "LEFT_EYE")
    reye = pt(lms, "RIGHT_EYE")

    nose   = pt(lms,"NOSE")

    ok = vis_ok

    # Precompute safe angles
    ang_le = angle(ls, le, lw) if all(map(ok,(ls,le,lw))) else 180
    ang_re = angle(rs, re, rw) if all(map(ok,(rs,re,rw))) else 180

    # 1) Tough Guy (arms crossed)
    if all(map(ok,(rw,ls,lw,rs))) and near(rw, ls, s, 0.45) and near(lw, rs, s, 0.45):
        return "Tough Guy Pose"

    # 2) Muscle Man (double biceps): hands above head/shoulders, elbows bent-ish, hands near own shoulders
    if all(map(ok,(lw, rw, le, re, ls, rs, nose))):
        wrists_high = (above(lw, ls, 0.05) or above(lw, nose, -0.02)) and \
                      (above(rw, rs, 0.05) or above(rw, nose, -0.02))
        elbows_bent = (50 <= ang_le <= 120) and (50 <= ang_re <= 120)
        # wrists_near_shoulders = near(lw, ls, s, 0.55) and near(rw, rs, s, 0.55)
        # if wrists_high and elbows_bent and wrists_near_shoulders:
        if wrists_high and elbows_bent:
            return "Muscle Man Pose"

    # 3) What? (shrug): hands around shoulder height, elbows bent a bit, shoulders high-ish
    if all(map(ok,(lw,rw,ls,rs,nose))):
        # wrists_at_sh = horiz_aligned(lw, ls, s) and horiz_aligned(rw, rs, s)
        left_in_band = between_vertical(lw, ls, le, s, top_eps=0.14, bot_eps=0.12)
        right_in_band = between_vertical(rw, rs, re, s, top_eps=0.14, bot_eps=0.12)
        elbows_bentish = (ang_le < 150 and ang_re < 150)
        # shoulders_high = (abs(ls.y - nose.y) < 0.12) or (abs(rs.y - nose.y) < 0.12)
        # if wrists_at_sh and elbows_bentish and shoulders_high:
        # if wrists_at_sh and elbows_bentish:
        if left_in_band and right_in_band and elbows_bentish:
            return "What? Pose"

    # 4) Point Up: one arm straight up; the other not up.
    up_l = all(map(ok,(lw, le, ls))) and above(lw, ls, 0.15) and ang_le > 150
    up_r = all(map(ok,(rw, re, rs))) and above(rw, rs, 0.15) and ang_re > 150
    if up_l ^ up_r:  # exactly one is True
        return "Point Up Pose (L)" if up_l else "Point Up Pose (R)"

    # 5) Samurai: one arm horizontal (extended) + other hand near hip
    arm_l_horizontal = all(map(ok,(ls,le,lw))) and ang_le > 150 and horiz_aligned(lw, ls, s, 0.10)
    arm_r_horizontal = all(map(ok,(rs,re,rw))) and ang_re > 150 and horiz_aligned(rw, rs, s, 0.10)
    hand_on_hip_l = all(map(ok,(lw,lh))) and near(lw, lh, s, 0.55)
    hand_on_hip_r = all(map(ok,(rw,rh))) and near(rw, rh, s, 0.55)
    if (arm_l_horizontal and hand_on_hip_r) or (arm_r_horizontal and hand_on_hip_l):
        return "Samurai Pose"

    # # 6) Mantis: both hands near face (we’ll ignore legs for now as you asked)
    # if all(map(ok,(lw,rw,nose,ls,rs))):
    #     # hands_near_face = near(lw, nose, s, 0.6) and near(rw, nose, s, 0.6) and dist(lw, rw) < 0.6*s
    #     hands_near_face = near(lw, nose, s, 0.6) or near(rw, nose, s, 0.6) and dist(lw, rw) < 0.6*s
    #     if hands_near_face:
    #         return "Mantis Pose"
    #
    # return ""
    # 6) Stop: one hand up near face (like saying "stop")
    if all(map(ok, (lw, rw, nose, ls, rs, leye, reye))):
        head_top_y = min(nose.y, leye.y, reye.y)  # visually higher = smaller y

        def is_stop(wrist, elbow, shoulder):
            # must be near the face (tighter), above its shoulder, but NOT way above the head
            close_to_face   = near(wrist, nose, s, 0.45)
            above_shoulder  = wrist.y < shoulder.y - 0.02 * s
            not_too_high    = wrist.y >= head_top_y - 0.12 * s   # keep within face band
            elbow_bentish   = angle(shoulder, elbow, wrist) < 150  # avoid straight-up arm (Point Up)
            # optional: ensure the hand is "in front" of the elbow (toward camera).
            # Mediapipe Pose z is camera-depth; typically "more negative" is closer to camera.
            forwardish      = (wrist.z < elbow.z - 0.03) if hasattr(wrist, "z") and hasattr(elbow, "z") else True

            # optional: keep it near face laterally too (reduce side triggers)
            lateral_ok      = abs(wrist.x - nose.x) <= 0.30 * s

            return close_to_face and above_shoulder and not_too_high and elbow_bentish and forwardish and lateral_ok

        left_stop  = all(map(ok, (lw, le, ls))) and is_stop(lw, le, ls)
        right_stop = all(map(ok, (rw, re, rs))) and is_stop(rw, re, rs)

        if left_stop or right_stop:
            return "Stop Pose"


# ---------- drawing ----------
def draw_and_label(frame_bgr, results, label, fps):
    if results.pose_landmarks:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_draw.draw_landmarks(
            rgb, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_style.get_default_pose_landmarks_style()
        )
        frame_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.putText(frame_bgr, f"FPS {fps:4.1f}", (8,24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(frame_bgr, f"Pose: {label or '(none)'}", (8,52), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,225,255), 2)
    return frame_bgr

# ---------- main ----------
def main():
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0, cv2.CAP_ANY)
    if not cap.isOpened():
        print("ERROR: could not open camera 0.")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    cv2.namedWindow("MediaPipe Pose – debug", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow("MediaPipe Pose – debug", 960, 540)

    pose = mp_pose.Pose(model_complexity=0)
    hist = deque(maxlen=10)
    last, last_change = "", 0.0
    COOLDOWN = 0.25

    prev = time.time()
    while True:
        ok, frame = cap.read()
        if not ok: break
        frame = cv2.flip(frame, 1)
        res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        label_now = ""
        if res.pose_landmarks:
            label_now = classify_pose(res.pose_landmarks.landmark)

        # stability: majority vote + debounce
        hist.append(label_now)
        voted = ""
        nonempty = [g for g in hist if g]
        if nonempty:
            voted = Counter(nonempty).most_common(1)[0][0]
        now = time.time()
        if voted != last and (now - last_change) >= COOLDOWN:
            last = voted
            last_change = now

        fps = 1.0 / max(1e-6, now - prev); prev = now
        disp = draw_and_label(frame, res, last, fps)
        cv2.imshow("MediaPipe Pose – debug", disp)
        if (cv2.waitKey(1) & 0xFF) == 27: break  # ESC

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


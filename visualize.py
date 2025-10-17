import cv2, time, math
import mediapipe as mp
from collections import deque, Counter

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# ---------------- helpers ----------------
M = mp_pose.PoseLandmark

def pt(lms, name):
    return lms[M[name].value]

def vis_ok(p, th=0.6):
    return (p.visibility or 0) >= th

def angle(a, b, c):
    # angle at b between segments ba and bc (in degrees)
    abx, aby = a.x - b.x, a.y - b.y
    cbx, cby = c.x - b.x, c.y - b.y
    dot = abx*cbx + aby*cby
    na = math.hypot(abx, aby); nc = math.hypot(cbx, cby)
    if na*nc == 0: return 180.0
    cosv = max(-1.0, min(1.0, dot/(na*nc)))
    return math.degrees(math.acos(cosv))

def dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)

def torso_scale(lms):
    # a size reference to make thresholds resolution-invariant
    ls, rs = pt(lms,"LEFT_SHOULDER"), pt(lms,"RIGHT_SHOULDER")
    lh, rh = pt(lms,"LEFT_HIP"), pt(lms,"RIGHT_HIP")
    if not all(map(vis_ok, (ls,rs,lh,rh))): return 0.3
    shoulder_w = dist(ls, rs)
    torso_h = dist(((ls)), ((lh))) + dist(((rs)), ((rh))) / 2.0
    return max(0.25, min(0.6, (shoulder_w + torso_h*0.5)))

# y goes DOWN in image space (smaller y = higher)
def above(p, q, margin=0.0): return p.y < q.y - margin
def near(a, b, s, k):        return dist(a,b) < k*s
def horiz_aligned(a, b, s, eps=0.12): return abs(a.y - b.y) < eps*s

# ------------- pose rules (heuristics) -------------
def detect_pose_name(lms):
    """Return one of: Muscle Man, What?, Point Up (L/R), Tough Guy, Samurai, Mantis, or ''."""
    s = torso_scale(lms)

    lw, rw = pt(lms,"LEFT_WRIST"), pt(lms,"RIGHT_WRIST")
    le, re = pt(lms,"LEFT_ELBOW"), pt(lms,"RIGHT_ELBOW")
    ls, rs = pt(lms,"LEFT_SHOULDER"), pt(lms,"RIGHT_SHOULDER")
    lh, rh = pt(lms,"LEFT_HIP"), pt(lms,"RIGHT_HIP")
    lk, rk = pt(lms,"LEFT_KNEE"), pt(lms,"RIGHT_KNEE")
    la, ra = pt(lms,"LEFT_ANKLE"), pt(lms,"RIGHT_ANKLE")
    nose   = pt(lms,"NOSE")

    ok = vis_ok

    # Angles (straight ≈ 180, right angle ≈ 90)
    ang_re = angle(rs, re, rw) if all(map(ok,(rs,re,rw))) else 180
    ang_le = angle(ls, le, lw) if all(map(ok,(ls,le,lw))) else 180
    ang_rs = angle(re, rs, ls) if all(map(ok,(re,rs,ls))) else 180
    ang_ls = angle(le, ls, rs) if all(map(ok,(le,ls,rs))) else 180

    # Feet spread (Samurai stance)
    feet_apart = (ok(ra) and ok(la) and dist(ra,la) > 1.3 * dist(ls,rs))

    # --- Tough Guy (arms crossed) ---
    if all(map(ok,(rw,ls,lw,rs))) and near(rw, ls, s, 0.45) and near(lw, rs, s, 0.45):
        return "Tough Guy Pose"

    # --- Muscle Man (double biceps): elbows ~90, wrists near head/shoulders ---
    if all(map(ok,(lw, rw, le, re, ls, rs))):
        elbows_bent = (50 <= ang_le <= 120) and (50 <= ang_re <= 120)
        wrists_high = ((ok(nose) and (above(lw, ls, 0.05) or above(lw, nose,-0.02))) and
                       (ok(nose) and (above(rw, rs, 0.05) or above(rw, nose,-0.02))))
        wrists_near_shoulders = near(lw, ls, s, 0.55) and near(rw, rs, s, 0.55)
        if elbows_bent and wrists_high and wrists_near_shoulders:
            return "Muscle Man Pose"

    # --- What? (shrug): wrists at shoulder height, shoulders “high” (close to ears) ---
    if all(map(ok,(lw,rw,ls,rs))):
        wrists_at_sh = horiz_aligned(lw, ls, s) and horiz_aligned(rw, rs, s)
        # shoulders close to head (approx via nose y)
        shoulders_high = ok(nose) and (abs(ls.y - nose.y) < 0.12 or abs(rs.y - nose.y) < 0.12)
        elbows_bentish = (ang_le < 140 and ang_re < 140)
        if wrists_at_sh and elbows_bentish and shoulders_high:
            return "What? Pose"

    # --- Point Up (one arm straight up, other free) ---
    if all(map(ok,(rw, re, rs))):
        up_r = above(rw, rs, 0.15) and ang_re > 150
    else:
        up_r = False
    if all(map(ok,(lw, le, ls))):
        up_l = above(lw, ls, 0.15) and ang_le > 150
    else:
        up_l = False
    if up_l and not up_r: return "Point Up Pose (L)"
    if up_r and not up_l: return "Point Up Pose (R)"
    if up_l and up_r:     return "Point Up Pose (Both)"  # optional

    # --- Samurai: wide feet + one arm extended ~horizontal, other near hip ---
    arm_l_horizontal = all(map(ok,(ls,le,lw))) and ang_le > 150 and horiz_aligned(lw, ls, s)
    arm_r_horizontal = all(map(ok,(rs,re,rw))) and ang_re > 150 and horiz_aligned(rw, rs, s)
    hand_on_hip_l = all(map(ok,(lw,lh))) and near(lw, lh, s, 0.55)
    hand_on_hip_r = all(map(ok,(rw,rh))) and near(rw, rh, s, 0.55)
    if feet_apart and ((arm_l_horizontal and hand_on_hip_r) or (arm_r_horizontal and hand_on_hip_l)):
        return "Samurai Pose"

    # --- Mantis: one knee lifted + hands near face (close together) ---
    knee_up_l = ok(lk) and ok(lh) and above(lk, lh, 0.10)
    knee_up_r = ok(rk) and ok(rh) and above(rk, rh, 0.10)
    hands_near_face = (ok(lw) and ok(rw) and ok(nose) and
                       near(lw, nose, s, 0.6) and near(rw, nose, s, 0.6) and
                       dist(lw, rw) < 0.6*s)
    if hands_near_face and (knee_up_l or knee_up_r):
        return "Mantis Pose"

    return ""  # unknown/neutral

# ------------- main loop -------------
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    pose = mp_pose.Pose(model_complexity=0)
    hist = deque(maxlen=10)
    last = ""
    last_change = 0.0
    COOLDOWN = 0.25

    prev = time.time()
    while True:
        ok, frame = cap.read()
        if not ok: break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

        label = ""
        if res.pose_landmarks:
            lms = res.pose_landmarks.landmark
            label = detect_pose_name(lms)

            # draw landmarks
            mp_draw.draw_landmarks(
                frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style()
            )

        # majority vote + debounce to keep label stable
        hist.append(label)
        voted = ""
        nonempty = [g for g in hist if g]
        if nonempty:
            voted = Counter(nonempty).most_common(1)[0][0]

        now = time.time()
        if voted != last and (now - last_change) >= COOLDOWN:
            last = voted
            last_change = now

        # overlay HUD
        fps = 1.0 / max(1e-6, now - prev); prev = now
        cv2.putText(frame, f"FPS {fps:4.1f}", (8,24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(frame, f"Pose: {last or '(none)'}", (8,52), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50,220,50), 2)

        cv2.imshow("MediaPipe Pose – debug", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


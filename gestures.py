import mediapipe as mp
mp_pose = mp.solutions.pose

def get_landmark(landmarks, name):
    return landmarks[mp_pose.PoseLandmark[name].value]

def distance(a, b):
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5

def detect_gestures(landmarks):
    output = []
    left_wrist  = get_landmark(landmarks, "LEFT_WRIST")
    right_wrist = get_landmark(landmarks, "RIGHT_WRIST")
    left_ear    = get_landmark(landmarks, "LEFT_EAR")
    right_ear   = get_landmark(landmarks, "RIGHT_EAR")
    left_sh     = get_landmark(landmarks, "LEFT_SHOULDER")
    right_sh    = get_landmark(landmarks, "RIGHT_SHOULDER")

    if distance(right_wrist, right_ear) < 0.15:
        print("right ear")
        output.append("Scratching head (R)")
    elif distance(left_wrist, left_ear) < 0.15:
        print("left ear")
        output.append("Scratching head (L)")

    if distance(right_wrist, left_sh) < 0.25 and distance(left_wrist, right_sh) < 0.25:
        print("arms crossed")
        output.append("Arms crossed")

    if distance(right_sh, right_wrist) < 0.10 and distance(left_sh, left_wrist) < 0.10:
        print("shrugging")
        output.append("Shrugging")

    if left_wrist.y < left_sh.y - 0.10:
        print("left raised")
        output.append("Left hand raised")
    if right_wrist.y < right_sh.y - 0.10:
        print("right raised")
        output.append("Right hand raised")

    return output


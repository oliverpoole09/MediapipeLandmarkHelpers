import cv2
import mediapipe as mp
import math

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cam = cv2.VideoCapture(1)

clicked_point = None
tracked_hand_idx = None
tracked_landmark_idx = None

cv2.namedWindow("Hand Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Hand Detection", 960, 720)

def on_mouse(event, x, y, flags, param):
    global clicked_point
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)

cv2.setMouseCallback("Hand Detection", on_mouse)

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=6,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

while True:
    success, frame = cam.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    h, w, _ = frame.shape

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        if clicked_point:
            cx, cy = clicked_point
            best_hand_idx = None
            best_lm_idx = None
            best_dist = float("inf")

            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                for i, lm in enumerate(hand_landmarks.landmark):
                    x, y = int(lm.x * w), int(lm.y * h)
                    dist = math.hypot(cx - x, cy - y)
                    if dist < best_dist:
                        best_dist = dist
                        best_hand_idx = hand_idx
                        best_lm_idx = i

            if best_dist < 15:
                tracked_hand_idx = best_hand_idx
                tracked_landmark_idx = best_lm_idx
            clicked_point = None

        if tracked_hand_idx is not None and tracked_landmark_idx is not None:
            if tracked_hand_idx < len(results.multi_hand_landmarks):
                hand_landmarks = results.multi_hand_landmarks[tracked_hand_idx]
                lm = hand_landmarks.landmark[tracked_landmark_idx]
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (x, y), 6, (0, 255, 0), -1)
                cv2.putText(frame, str(tracked_landmark_idx), (x + 8, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Hand Detection", frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cam.release()
cv2.destroyAllWindows()

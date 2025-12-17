import cv2
import mediapipe as mp
import math

face_mesh_sol = mp.solutions.face_mesh
drawing_sol = mp.solutions.drawing_utils

cam = cv2.VideoCapture(1)

clicked_point = None
tracked_landmark_idx = None

cv2.namedWindow("Webcam", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Webcam", 960, 720)

def on_mouse(event, x, y, flags, param):
    global clicked_point
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)

cv2.setMouseCallback("Webcam", on_mouse)

with face_mesh_sol.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:
    while cam.isOpened():
        ret, frame = cam.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb_frame)
        h, w, _ = frame.shape

        if result.multi_face_landmarks:
            face_landmarks = result.multi_face_landmarks[0]
            drawing_sol.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=face_mesh_sol.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawing_sol.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1),
                connection_drawing_spec=drawing_sol.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1)
            )

            if clicked_point:
                cx, cy = clicked_point
                min_dist = float("inf")
                min_index = None

                for i, lm in enumerate(face_landmarks.landmark):
                    x, y = int(lm.x * w), int(lm.y * h)
                    dist = math.hypot(cx - x, cy - y)
                    if dist < min_dist:
                        min_dist = dist
                        min_index = i

                if min_dist < 10:
                    tracked_landmark_idx = min_index
                clicked_point = None

            if tracked_landmark_idx is not None:
                lm = face_landmarks.landmark[tracked_landmark_idx]
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(frame, str(tracked_landmark_idx), (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Webcam", frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cam.release()
cv2.destroyAllWindows()
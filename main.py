import cv2
import mediapipe as mp
import numpy as np
import time
import pygame
from threading import Thread
import os

# ================== ÂM THANH ==================
wav_path = "alarm.wav"

def play_sound():
    base = os.path.dirname(os.path.abspath(__file__))
    sound = os.path.join(base, wav_path)
    pygame.mixer.init()
    pygame.mixer.music.load(sound)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        time.sleep(0.1)

# ================== HÀM TÍNH ==================
def dist(a, b):
    return np.linalg.norm(a - b)

def eye_ratio(eye):
    return (dist(eye[1], eye[5]) + dist(eye[2], eye[4])) / (2.0 * dist(eye[0], eye[3]))

def mouth_ratio(m):
    return dist(m[2], m[10]) / dist(m[0], m[6])

# ================== NGƯỠNG CỐ ĐỊNH ==================
EYE_THR      = 0.23     # ngưỡng nhắm mắt
MAX_SLEEP    = 25       # số frame nhắm mắt liên tục

YAWN_THR     = 0.23     # ngưỡng ngáp
YAWN_FRAMES  = 10       # số frame ngáp liên tục

# ================== BIẾN ĐẾM ==================
sleep_frames = 0
yawn_frames  = 0
alarmed = False

# ================== MEDIAPIPE ==================
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH     = [78, 81, 13, 311, 308, 402, 14, 178, 87, 317, 82, 312]

# ================== CAMERA ==================
cap = cv2.VideoCapture(0)
time.sleep(1)

cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Camera", 800, 600)

# ================== MAIN LOOP ==================
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:
        h, w, _ = frame.shape
        lm = result.multi_face_landmarks[0].landmark

        leftEye  = np.array([[lm[i].x * w, lm[i].y * h] for i in LEFT_EYE])
        rightEye = np.array([[lm[i].x * w, lm[i].y * h] for i in RIGHT_EYE])
        mouth    = np.array([[lm[i].x * w, lm[i].y * h] for i in MOUTH])

        ear = (eye_ratio(leftEye) + eye_ratio(rightEye)) / 2.0
        mar = mouth_ratio(mouth)

        # ====== HIỂN THỊ GIÁ TRỊ ======
        cv2.putText(frame, f"EYE: {ear:.3f}", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(frame, f"MOUTH: {mar:.3f}", (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # ====== TRẠNG THÁI NGÁP (KHÔNG CẢNH BÁO) ======
        if mar > YAWN_THR:
            yawn_frames += 1
            if yawn_frames >= YAWN_FRAMES:
                cv2.putText(frame, "BAN DANG NGAP", (30, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 255), 3)
        else:
            yawn_frames = 0

        # ====== CẢNH BÁO BUỒN NGỦ (NHẮM MẮT) ======
        if ear < EYE_THR:
            sleep_frames += 1
            if sleep_frames >= MAX_SLEEP:
                if not alarmed:
                    alarmed = True
                    Thread(target=play_sound, daemon=True).start()

                cv2.putText(frame, "CANH BAO BUON NGU!", (30, 190),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255), 3)
        else:
            sleep_frames = 0
            alarmed = False

    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# ================== CLEAN ==================
cap.release()
cv2.destroyAllWindows()
pygame.quit()

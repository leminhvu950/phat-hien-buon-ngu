from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import imutils
import time
from threading import Thread
import dlib
import os
import cv2
import pygame  # dùng pygame để phát âm thanh

# Đường dẫn đến file âm thanh
wav_path = "alarm.wav"

# Hàm phát ra âm thanh
def play_sound(path):
    base_path = os.path.dirname(os.path.abspath(__file__))  # thư mục chứa file .py
    sound_path = os.path.join(base_path, path)

    pygame.mixer.init()
    pygame.mixer.music.load(sound_path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        continue

# Hàm tính khoảng cách giữa 2 điểm
def e_dist(pA, pB):
    return np.linalg.norm(pA - pB)

# Tính tỷ lệ mắt
def eye_ratio(eye):
    d_V1 = e_dist(eye[1], eye[5])
    d_V2 = e_dist(eye[2], eye[4])
    d_H = e_dist(eye[0], eye[3])
    return (d_V1 + d_V2) / (2.0 * d_H)

# Tính tỷ lệ ngáp (miệng)
def mouth_ratio(mouth):
    d_V = e_dist(mouth[2], mouth[10])  # Khoảng cách dọc (giữa môi trên và môi dưới)
    d_H = e_dist(mouth[0], mouth[6])   # Khoảng cách ngang (giữa hai mép miệng)
    return d_V / d_H

# Ngưỡng tỷ lệ mắt để xác định buồn ngủ
eye_ratio_threshold = 0.2
# Ngưỡng tỷ lệ miệng để xác định ngáp
yawn_ratio_threshold = 0.8

# Threshold số frame liên tục nhắm mắt/ngáp
max_sleep_frames = 45
sleep_frames = 0

# Check xem đã cảnh báo hay chưa
alarmed = False

# Khởi tạo các module detect mặt và facial landmark
face_detect = cv2.CascadeClassifier(r"D:\haarcascade_frontalface_default.xml")
landmark_detect = dlib.shape_predictor(r"D:\shape_predictor_68_face_landmarks.dat")

# Lấy danh sách các cụm điểm landmark cho 2 mắt và miệng
(left_eye_start, left_eye_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(right_eye_start, right_eye_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mouth_start, mouth_end) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

# Đọc từ camera
vs = VideoStream(src=0).start()
time.sleep(1.0)

while True:
    # Đọc từ camera
    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect các mặt trong ảnh
    faces = face_detect.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, 
        minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Duyệt qua các mặt
    for (x, y, w, h) in faces:
        rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))

        # Nhận diện các điểm landmark
        landmark = landmark_detect(gray, rect)
        landmark = face_utils.shape_to_np(landmark)

        # Tính toán tỷ lệ mắt trái, mắt phải, và trung bình
        leftEye = landmark[left_eye_start:left_eye_end]
        rightEye = landmark[right_eye_start:right_eye_end]
        left_eye_ratio = eye_ratio(leftEye)
        right_eye_ratio = eye_ratio(rightEye)
        eye_avg_ratio = (left_eye_ratio + right_eye_ratio) / 2.0

        # Tính toán tỷ lệ miệng để phát hiện ngáp
        mouth = landmark[mouth_start:mouth_end]
        mouth_ratio_val = mouth_ratio(mouth)

        # Vẽ đường bao quanh mắt và miệng
        left_eye_bound = cv2.convexHull(leftEye)
        right_eye_bound = cv2.convexHull(rightEye)
        mouth_bound = cv2.convexHull(mouth)

        cv2.drawContours(frame, [left_eye_bound], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [right_eye_bound], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [mouth_bound], -1, (0, 255, 0), 1)

        # Check xem mắt có nhắm không
        if eye_avg_ratio < eye_ratio_threshold:
            sleep_frames += 1
            if sleep_frames >= max_sleep_frames:
                if not alarmed:
                    alarmed = True
                    t = Thread(target=play_sound, args=(wav_path,))
                    t.daemon = True  # sửa chính tả
                    t.start()
                cv2.putText(frame, "CANH BAO BUON NGU!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            sleep_frames = 0
            alarmed = False
            cv2.putText(frame, "EYE AVG RATIO: {:.3f}".format(eye_avg_ratio),
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Check xem có ngáp không
        if mouth_ratio_val > yawn_ratio_threshold:
            cv2.putText(frame, "CANH BAO NGAP!", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "MOUTH RATIO: {:.3f}".format(mouth_ratio_val),
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Hiển thị lên màn hình
    cv2.imshow("Camera", frame)

    # Bấm Esc để thoát
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cv2.destroyAllWindows()
vs.stop()

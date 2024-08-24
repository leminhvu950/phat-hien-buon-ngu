import cv2
import dlib
import numpy as np
import time

# Khởi tạo máy dò khuôn mặt và máy dò các điểm đặc trưng trên khuôn mặt của Dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r"D:\shape_predictor_68_face_landmarks.dat")  # Đảm bảo bạn đã tải tệp này

# Mở camera
cap = cv2.VideoCapture(0)

# Định nghĩa ngưỡng cho ngáp và nhắm mắt
MOUTH_OPEN_THRESHOLD = 25  # Điều chỉnh ngưỡng này nếu cần
EAR_THRESHOLD = 0.25  # Ngưỡng cho tỷ lệ mắt

# Biến lưu thời gian nhắm mắt
eyes_closed_start_time = None

def calculate_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

def calculate_eye_aspect_ratio(eye_points):
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    C = np.linalg.norm(eye_points[0] - eye_points[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Tạo cửa sổ và đặt chế độ toàn màn hình
cv2.namedWindow("Face, Eyes, and Mouth Detection", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Face, Eyes, and Mouth Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    # Đọc khung hình từ camera
    ret, frame = cap.read()
    if not ret:
        break

    # Chuyển đổi khung hình sang thang độ xám (grayscale)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Phát hiện khuôn mặt trong khung hình
    faces = detector(gray)

    for face in faces:
        # Phát hiện các điểm đặc trưng trên khuôn mặt
        landmarks = predictor(gray, face)

        # Lấy điểm đặc trưng cho môi
        mouth_top = (landmarks.part(51).x, landmarks.part(51).y)  # Điểm trên môi
        mouth_bottom = (landmarks.part(57).x, landmarks.part(57).y)  # Điểm dưới môi

        # Tính toán khoảng cách giữa hai điểm môi
        mouth_distance = calculate_distance(mouth_top, mouth_bottom)

        # Kiểm tra ngáp
        if mouth_distance > MOUTH_OPEN_THRESHOLD:
            cv2.putText(frame, "Yawning", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Tính toán tỷ lệ mắt và kiểm tra nhắm mắt
        left_eye_points = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
        right_eye_points = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])
        
        left_eye_ear = calculate_eye_aspect_ratio(left_eye_points)
        right_eye_ear = calculate_eye_aspect_ratio(right_eye_points)

        if left_eye_ear < EAR_THRESHOLD and right_eye_ear < EAR_THRESHOLD:
            cv2.putText(frame, "Eyes Closed", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            if eyes_closed_start_time is None:
                eyes_closed_start_time = time.time()
            elif time.time() - eyes_closed_start_time >= 2:  # Kiểm tra nếu đã nhắm mắt được 2 giây
                cv2.putText(frame, "Sleeping", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            eyes_closed_start_time = None

        # Vẽ các điểm đặc trưng trên môi
        cv2.circle(frame, mouth_top, 2, (255, 0, 0), -1)  # Môi trên
        cv2.circle(frame, mouth_bottom, 2, (255, 0, 0), -1)  # Môi dưới

        # Vẽ các điểm đặc trưng trên mắt
        for (x, y) in left_eye_points:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)  # Mắt trái
        for (x, y) in right_eye_points:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)  # Mắt phải

        # Vẽ hình chữ nhật xung quanh khuôn mặt
        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)

    # Hiển thị khung hình với các khuôn mặt, mắt và miệng được đánh dấu
    cv2.imshow("Face, Eyes, and Mouth Detection", frame)

    # Nhấn phím 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()

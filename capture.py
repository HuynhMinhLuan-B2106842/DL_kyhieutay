import cv2
import mediapipe as mp
import os
import time

# Khởi tạo thư viện MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

nameoflable = 'Duong_C'

# Tạo thư mục để lưu ảnh
output_dir = f'hand_images/{nameoflable}'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Khởi tạo webcam
cap = cv2.VideoCapture(0)

# Đếm số lượng ảnh chụp
img_count = 0
max_images = 200

# Tỷ lệ mở rộng vùng chứa bàn tay
padding = 0.5  # Giá trị này sẽ thêm viền ngoài vào ảnh, có thể điều chỉnh


while img_count < max_images:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Chuyển khung hình sang RGB vì MediaPipe yêu cầu định dạng này
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Phát hiện bàn tay
    result = hands.process(frame_rgb)
    
    # Nếu phát hiện bàn tay
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # # Vẽ các điểm và kết nối bàn tay (tùy chọn)
            # mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Lấy bounding box của bàn tay
            h, w, c = frame.shape
            x_min = min([landmark.x for landmark in hand_landmarks.landmark]) * w
            x_max = max([landmark.x for landmark in hand_landmarks.landmark]) * w
            y_min = min([landmark.y for landmark in hand_landmarks.landmark]) * h
            y_max = max([landmark.y for landmark in hand_landmarks.landmark]) * h
            
            # Tính toán thêm viền ngoài (padding)
            box_width = x_max - x_min
            box_height = y_max - y_min
            x_min = max(0, int(x_min - padding * box_width))
            x_max = min(w, int(x_max + padding * box_width))
            y_min = max(0, int(y_min - padding * box_height))
            y_max = min(h, int(y_max + padding * box_height))
            
            # Cắt khung hình rộng hơn chứa bàn tay và viền ngoài
            hand_image = frame[y_min:y_max, x_min:x_max]
            
            # Thay đổi kích thước ảnh về 256x256 để có chất lượng tốt hơn
            hand_image_resized = cv2.resize(hand_image, (128, 128))
            
            # Lưu ảnh vào thư mục
            img_filename = os.path.join(output_dir, f'{nameoflable}_{img_count}.jpg')
            cv2.imwrite(img_filename, hand_image_resized)
            
            # Tăng số đếm
            img_count += 1
            print(f'Đã chụp {img_count}/{max_images} ảnh')
    
    # Hiển thị khung hình
    cv2.imshow('Hand Detection', frame)
    
    # Thoát khi nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()

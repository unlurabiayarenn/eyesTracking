import cv2
import mediapipe as mp
import numpy as np

# Mediapipe ayarları
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Göz işaretleri için indeksler
LEFT_EYE = [362, 385, 387, 263, 373, 380,382]
RIGHT_EYE = [133,158,159,160, 158,153,154, 155, 144,145]

#LEFT_EYE = [382, 381, 374, 390, 386]  # Sol göz
#right_ EYE = [142]  # Sağ göz hatalı nokta

# Video akışını başlat
cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1) as face_mesh:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Yüz noktalarını çiz
                mp_drawing.draw_landmarks(
                    frame,
                    face_landmarks,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1)
                )

                # Göz noktalarını al
                left_eye = np.array([[face_landmarks.landmark[i].x * frame.shape[1],
                                       face_landmarks.landmark[i].y * frame.shape[0]] for i in LEFT_EYE], dtype=np.int32)
                right_eye = np.array([[face_landmarks.landmark[i].x * frame.shape[1],
                                        face_landmarks.landmark[i].y * frame.shape[0]] for i in RIGHT_EYE], dtype=np.int32)

                # Göz çevresindeki noktaları kırmızı yap
                for point in left_eye:
                    cv2.circle(frame, tuple(point), 3, (0, 0, 255), -1)  # Kırmızı
                for point in right_eye:
                    cv2.circle(frame, tuple(point), 3, (0, 0, 255), -1)  # Kırmızı

        cv2.imshow("Yuz ve Goz Isaretleme", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC tuşuna basınca çık
            break

cap.release()
cv2.destroyAllWindows()

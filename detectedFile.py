import cv2 as cv
import mediapipe as mp
import time
import colorFile, math
import numpy as np
import pandas as pd

# Değişkenler
frame_counter = 0
data = []  # Veriyi toplamak için bir liste oluştur
# Sabitler
CLOSED_EYES_FRAME = 3
FONTS = cv.FONT_HERSHEY_COMPLEX

# Yüz kenarları indeksleri
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

map_face_mesh = mp.solutions.face_mesh
# Kamera nesnesi (dahili kamera için 0 kullanılıyor)
camera = cv.VideoCapture(0)

# Landmark tespiti fonksiyonu
def landmarksDetection(img, results, draw=False):
    img_height, img_width = img.shape[:2]
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in
                  results.multi_face_landmarks[0].landmark]
    if draw:
        [cv.circle(img, p, 2, (0, 255, 0), -1) for p in mesh_coord]
    return mesh_coord

# Euclidean mesafe
def euclideanDistance(point, point1):
    x, y = point
    x1, y1 = point1
    distance = math.sqrt((x1 - x) ** 2 + (y1 - y) ** 2)
    return distance

# Göz durumu belirleme
def eyeStatus(landmarks, eye_indices):
    # Yatay ve dikey mesafeleri hesapla
    horizontal = euclideanDistance(landmarks[eye_indices[0]], landmarks[eye_indices[8]])
    vertical = euclideanDistance(landmarks[eye_indices[12]], landmarks[eye_indices[4]])
    ratio = horizontal / vertical
    return ratio, horizontal, vertical  # Oran, yatay ve dikey değerlerini döndür

with map_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    start_time = time.time()
    while True:
        frame_counter += 1
        ret, frame = camera.read()
        if not ret:
            break

        frame = cv.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv.INTER_CUBIC)
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            mesh_coords = landmarksDetection(frame, results, False)
            left_eye_ratio, left_eye_horizontal, left_eye_vertical = eyeStatus(mesh_coords, LEFT_EYE)
            right_eye_ratio, right_eye_horizontal, right_eye_vertical = eyeStatus(mesh_coords, RIGHT_EYE)

            # Oranların eşik değerlerini kontrol et
            left_eye_status = "Acik" if left_eye_ratio < 4.0 else "Kapali"  # Göz açıkken oran düşük
            right_eye_status = "Acik" if right_eye_ratio < 4.0 else "Kapali"  # Göz açıkken oran düşük

            # Veriyi data listesine ekleyin
            data.append([left_eye_ratio, right_eye_ratio, left_eye_status, right_eye_status])

            # Ekranda göster
            colorFile.colorBackgroundText(frame, f'Sol Göz: {left_eye_status}', FONTS, 0.7, (30, 100), 2, colorFile.PINK, colorFile.YELLOW)
            colorFile.colorBackgroundText(frame, f'Sag Göz: {right_eye_status}', FONTS, 0.7, (30, 150), 2, colorFile.PINK, colorFile.YELLOW)
            colorFile.colorBackgroundText(frame, f'Sol Göz Degeri: {round(left_eye_ratio, 2)}', FONTS, 0.7, (30, 200), 2, colorFile.GREEN, colorFile.YELLOW)
            colorFile.colorBackgroundText(frame, f'Sag Göze Degeri: {round(right_eye_ratio, 2)}', FONTS, 0.7, (30, 250), 2, colorFile.GREEN, colorFile.YELLOW)

            # Göz noktalarını tek tek çizin
            for point in LEFT_EYE:
                cv.circle(frame, mesh_coords[point], 2, colorFile.GREEN, -1, cv.LINE_AA)
            for point in RIGHT_EYE:
                cv.circle(frame, mesh_coords[point], 2, colorFile.GREEN, -1, cv.LINE_AA)

        end_time = time.time() - start_time

        cv.imshow('frame', frame)
        key = cv.waitKey(2)
        if key == ord('q') or key == ord('Q'):
            break

    # Kamera işlemleri bittikten sonra veriyi CSV dosyasına kaydet
    df = pd.DataFrame(data, columns=["LeftEyeRatio", "RightEyeRatio", "LeftEyeStatus", "RightEyeStatus"])
    df.to_csv("eye_data.csv", index=False)
    print("Göz oranları 'eye_data.csv' dosyasına kaydedildi.")

    cv.destroyAllWindows()
    camera.release()

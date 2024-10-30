import cv2 as cv
import mediapipe as mp
import time
import colorFile, math
import numpy as np

# değişkenler
frame_counter = 0
CEF_COUNTER_LEFT = 0
CEF_COUNTER_RIGHT = 0
# sabitler
CLOSED_EYES_FRAME = 3
FONTS = cv.FONT_HERSHEY_COMPLEX

# yüz kenarları indeksleri
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

map_face_mesh = mp.solutions.face_mesh
# kamera nesnesi (dahili kamera için 0 kullanılıyor)
camera = cv.VideoCapture(0)

# landmark tespiti fonksiyonu
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
    # yatay ve dikey mesafeleri hesapla
    horizontal = euclideanDistance(landmarks[eye_indices[0]], landmarks[eye_indices[8]])
    vertical = euclideanDistance(landmarks[eye_indices[12]], landmarks[eye_indices[4]])
    ratio = horizontal / vertical
    return ratio

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
            left_eye_ratio = eyeStatus(mesh_coords, LEFT_EYE)
            right_eye_ratio = eyeStatus(mesh_coords, RIGHT_EYE)

            left_eye_status = "Open" if left_eye_ratio > 5.5 else "Closed"
            right_eye_status = "Open" if right_eye_ratio > 5.5 else "Closed"

            colorFile.colorBackgroundText(frame, f'Left Eye: {left_eye_status}', FONTS, 0.7, (30, 100), 2, colorFile.PINK, colorFile.YELLOW)
            colorFile.colorBackgroundText(frame, f'Right Eye: {right_eye_status}', FONTS, 0.7, (30, 150), 2, colorFile.PINK, colorFile.YELLOW)

            cv.polylines(frame, [np.array([mesh_coords[p] for p in LEFT_EYE], dtype=np.int32)], True, colorFile.GREEN, 1, cv.LINE_AA)
            cv.polylines(frame, [np.array([mesh_coords[p] for p in RIGHT_EYE], dtype=np.int32)], True, colorFile.GREEN, 1, cv.LINE_AA)

        end_time = time.time() - start_time
        fps = frame_counter / end_time
        frame = colorFile.textWithBackground(frame, f'FPS: {round(fps, 1)}', FONTS, 1.0, (30, 50), bgOpacity=0.9, textThickness=2)

        cv.imshow('frame', frame)
        key = cv.waitKey(2)
        if key == ord('q') or key == ord('Q'):
            break

    cv.destroyAllWindows()
    camera.release()

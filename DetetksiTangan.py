import cv2
import mediapipe as mp
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Inisialisasi modul MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Buka kamera
cap = cv2.VideoCapture(0)

# Inisialisasi objek volume
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# Mendapatkan rentang volume
min_vol, max_vol, _ = volume.GetVolumeRange()

# Variabel untuk normalisasi jarak
prev_distance = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=5))

            # Mendeteksi landmark yang mewakili ujung jari tangan
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            # Menghitung jarak antara ujung ibu jari dan telunjuk
            distance = abs(thumb_tip.x - index_tip.x) + abs(thumb_tip.y - index_tip.y)

            # Normalisasi jarak untuk respons yang lebih baik
            if prev_distance is None:
                prev_distance = distance
            else:
                distance = (distance + prev_distance) / 2
                prev_distance = distance

            # Mengonversi jarak menjadi perubahan volume
            # Sesuaikan rentang dan sensitivitas sesuai kebutuhan
            volume_level = int((distance * 100) / 0.3)
            volume_level = min(max(volume_level, 0), 100)  

            # Mengatur volume speaker
            new_volume = ((volume_level / 100) * (max_vol - min_vol)) + min_vol
            volume.SetMasterVolumeLevel(new_volume, None)

            # Menampilkan indikator volume di layar
            cv2.putText(frame, f"Volume: {volume_level}%", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                        cv2.LINE_AA)

    cv2.imshow('Hand Detection', frame)

    key = cv2.waitKey(1)
    if key == 27:  # Tekan 'ESC' untuk keluar
        break

# Tutup kamera dan jendela OpenCV
cap.release()
cv2.destroyAllWindows()

import cv2
import mediapipe as mp
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Inisialisasi modul MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Inisialisasi Haar Cascade Classifier untuk deteksi wajah
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Buka kamera
cap = cv2.VideoCapture(0)

# Inisialisasi objek volume
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# Mendapatkan rentang volume
min_vol, max_vol, _ = volume.GetVolumeRange()

recording = False
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Deteksi wajah
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Deteksi tangan menggunakan MediaPipe
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

            # Mengatur volume berdasarkan jarak ibu jari dan telunjuk
            volume_level = int((distance) * 100)  # Menghitung level volume

            # Mengatur volume
            new_volume = ((volume_level / 100) * (max_vol - min_vol)) + min_vol
            volume.SetMasterVolumeLevel(new_volume, None)

            # Menghitung tinggi bar berdasarkan level volume yang diatur
            bar_height = int(100 * ((new_volume - min_vol) / (max_vol - min_vol)))  # Ubah batas maksimum tinggi bar

            # Menampilkan bar volume secara vertikal dengan posisi yang lebih rendah
            cv2.rectangle(frame, (50, 280), (80, 280 - bar_height), (50, 50, 50), 3)  # Garis batas bar volume
            cv2.rectangle(frame, (52, 280 - bar_height), (78, 280), (0, 255, 0), -1)  # Bar volume dinamis
            cv2.rectangle(frame, (52, 280 - bar_height), (78, 280), (255, 255, 255), 1)  # Garis tepi bar volume

            # Menampilkan indikator level volume di dalam bar
            cv2.putText(frame, f"{volume_level}%", (60, 290 - bar_height), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)

    # Menampilkan kotak deteksi wajah
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Mulai merekam saat tombol spasi ditekan
    key = cv2.waitKey(1)
    if key == ord(' '):
        if not recording:
            recording = True
            out = cv2.VideoWriter('output.avi', fourcc, 20.0, (frame.shape[1], frame.shape[0]))
        else:
            recording = False
            out.release()

    if recording:
        out.write(frame)

    cv2.imshow('Hand and Face Detection', frame)

    if key == 27:  # Tekan 'ESC' untuk keluar
        break

# Tutup kamera dan jendela OpenCV
cap.release()
cv2.destroyAllWindows()

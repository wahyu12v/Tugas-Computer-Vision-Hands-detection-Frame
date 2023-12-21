import cv2
import mediapipe as mp
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

min_vol, max_vol, _ = volume.GetVolumeRange()

prev_distance = None

record_video = False
video_writer = None

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

            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            distance = abs(thumb_tip.x - index_tip.x) + abs(thumb_tip.y - index_tip.y)

            if prev_distance is None:
                prev_distance = distance
            else:
                distance = (distance + prev_distance) / 2
                prev_distance = distance

            volume_level = int((distance * 100) / 0.3)
            volume_level = min(max(volume_level, 0), 100)

            new_volume = ((volume_level / 100) * (max_vol - min_vol)) + min_vol
            volume.SetMasterVolumeLevel(new_volume, None)

            cv2.putText(frame, f"Volume: {volume_level}%", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                        cv2.LINE_AA)

    # Deteksi wajah
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Merekam video ketika spasi ditekan
    key = cv2.waitKey(1)
    if key == 27: 
        break
    elif key == 32:  
        record_video = not record_video
        if record_video:
            video_writer = cv2.VideoWriter('captured_video.avi', cv2.VideoWriter_fourcc(*'XVID'), 20.0,
                                           (frame.shape[1], frame.shape[0]))
        else:
            if video_writer is not None:
                video_writer.release()
                video_writer = None

    # Jika wajah terdeteksi, gambar kotak di sekitar wajah
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Menulis frame ke video jika sedang direkam
    if record_video and video_writer is not None:
        video_writer.write(frame)

    cv2.imshow('Hand Detection', frame)

# Tutup kamera dan jendela OpenCV
cap.release()
cv2.destroyAllWindows()

import cv2
import mediapipe as mp

# Inisialisasi modul Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Buka kamera
cap = cv2.VideoCapture(0)

count = 0  # Variabel penomoran untuk nama file

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Ubah warna frame menjadi RGB karena Mediapipe menggunakan format RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Proses deteksi tangan menggunakan Mediapipe
    results = hands.process(image)

    # Gambar landmark jika tangan terdeteksi
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Tampilkan frame
    cv2.imshow('Hand Detection', frame)

    # Menyimpan gambar ketika tombol spasi ditekan
    key = cv2.waitKey(1)
    if key == 32:  # ASCII untuk spasi
        filename = f'hand_{count}.png'
        cv2.imwrite(filename, frame)
        print(f"Image {filename} saved")
        count += 1

    # Tekan 'ESC' untuk keluar
    if key == 27:
        break

# Tutup kamera dan jendela OpenCV
cap.release()
cv2.destroyAllWindows()

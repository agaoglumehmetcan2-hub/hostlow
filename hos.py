import datetime
import math
import os
import threading
import time

import cv2
import mediapipe as mp

# --------------------------- Ayarlar ---------------------------

CAMERA_INDEX = 0

NORMAL_COMMAND_HOLD_SECONDS = 3.0
NURSE_CALL_HOLD_SECONDS = 4.0

EAR_CLOSED_THRESHOLD = 0.20
EYE_CLOSED_ALERT_SECONDS = 2.0

HEAD_DOWN_PITCH_THRESHOLD = 18
HEAD_DOWN_ALERT_SECONDS = 4.0

LOG_DIR = "hospital_logs"
SNAPSHOT_DIR = "hospital_captures"
EVENT_LOG_FILE = os.path.join(LOG_DIR, "events.log")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

ENABLE_TTS = True

# --------------------------- Sesli asistan ---------------------------

def speak(text):
    if not ENABLE_TTS:
        return

    def run_speech():
        try:
            import pyttsx3

            engine = pyttsx3.init()
            engine.setProperty("rate", 165)
            engine.say(text)
            engine.runAndWait()
        except Exception as error:
            print(f"[SES HATASI] {error}")

    threading.Thread(target=run_speech, daemon=True).start()

# --------------------------- Loglama ---------------------------

def write_log(message):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"

    try:
        with open(EVENT_LOG_FILE, "a", encoding="utf-8") as file:
            file.write(line + "\n")
    except Exception as error:
        print(f"[LOG HATASI] {error}")

    print(line)

# --------------------------- MediaPipe modülleri ---------------------------

mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)

# --------------------------- Kamera ---------------------------

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

if not cap.isOpened():
    cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print("[HATA] Kamera açılamadı.")
    write_log("HATA: Kamera açılamadı.")
    exit()

# --------------------------- Durum değişkenleri ---------------------------

system_state = "READY"
last_state = None

active_gesture = None
gesture_start_time = None
nurse_call_sent = False

light_on = False

eye_closed_start = None
head_down_start = None

alert_eye = False
alert_head = False

# --------------------------- Yardımcı fonksiyonlar ---------------------------

def count_fingers(landmarks):
    fingers = 0

    finger_pairs = [
        (8, 6),
        (12, 10),
        (16, 14),
        (20, 18),
    ]

    for tip_id, pip_id in finger_pairs:
        if landmarks[tip_id].y < landmarks[pip_id].y:
            fingers += 1

    return fingers

def euclidean_distance(point1, point2):
    return math.hypot(point1[0] - point2[0], point1[1] - point2[1])

def calculate_ear(landmarks, eye_indices, frame_w, frame_h):
    points = []

    for index in eye_indices:
        point = landmarks[index]
        points.append((point.x * frame_w, point.y * frame_h))

    vertical_1 = euclidean_distance(points[1], points[5])
    vertical_2 = euclidean_distance(points[2], points[4])
    horizontal = euclidean_distance(points[0], points[3])

    if horizontal == 0:
        return 0.0

    return (vertical_1 + vertical_2) / (2.0 * horizontal)

def estimate_head_pose(landmarks, frame_w, frame_h):
    nose = landmarks[1]
    forehead = landmarks[10]
    chin = landmarks[152]

    nose_y = nose.y * frame_h
    forehead_y = forehead.y * frame_h
    chin_y = chin.y * frame_h

    face_height = abs(chin_y - forehead_y)

    if face_height == 0:
        return 0.0

    # Basit pitch tahmini: yüz merkezine göre burun konumu
    pitch = ((nose_y - frame_h / 2) / (frame_h / 2)) * 45

    return pitch

def draw_hud(image, text, position, text_color=(20, 20, 20), background=(235, 235, 235)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1

    text_size, baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = position

    cv2.rectangle(
        image,
        (x - 6, y - text_size[1] - 8),
        (x + text_size[0] + 6, y + baseline + 6),
        background,
        -1,
    )

    cv2.putText(image, text, (x, y), font, scale, text_color, thickness)

def save_snapshot(frame, event_name):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = os.path.join(SNAPSHOT_DIR, f"{event_name}_{timestamp}.jpg")

    try:
        cv2.imwrite(file_name, frame)
        write_log(f"{event_name} için görüntü kaydedildi: {file_name}")
        return file_name
    except Exception as error:
        write_log(f"Görüntü kaydedilemedi: {error}")
        return None

# --------------------------- Başlatma ---------------------------

write_log("Hastane temassız kontrol ve dikkat izleme paneli başlatıldı.")
speak("Hastane kontrol paneli hazır.")

previous_frame_time = time.time()

# --------------------------- Ana döngü ---------------------------

while cap.isOpened():
    loop_start_time = time.time()

    success, frame = cap.read()

    if not success:
        print("[HATA] Kamera görüntüsü alınamadı.")
        write_log("HATA: Kamera görüntüsü alınamadı.")
        break

    frame = cv2.flip(frame, 1)
    frame_h, frame_w = frame.shape[:2]

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    hand_results = hands.process(rgb_frame)
    face_results = face_mesh.process(rgb_frame)

    current_time = datetime.datetime.now().strftime("%H:%M:%S")
    now = time.time()

    fps = 0
    frame_duration = now - previous_frame_time

    if frame_duration > 0:
        fps = int(1 / frame_duration)

    previous_frame_time = now

    status_text = "SISTEM HAZIR"
    detail_text = "Bekleme modu"
    state_color = (180, 180, 180)

    detected_fingers = 0

    # --------------------------- El hareketi ---------------------------

    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
            )

            landmarks = hand_landmarks.landmark
            detected_fingers = count_fingers(landmarks)

    if detected_fingers > 0:
        if active_gesture != detected_fingers:
            active_gesture = detected_fingers
            gesture_start_time = now
            nurse_call_sent = False

        required_time = (
            NURSE_CALL_HOLD_SECONDS
            if detected_fingers == 4
            else NORMAL_COMMAND_HOLD_SECONDS
        )

        elapsed = now - gesture_start_time
        remaining = max(0.0, required_time - elapsed)

        if elapsed >= required_time:
            if detected_fingers == 1:
                system_state = "SCREEN_LOCK"
                status_text = "EKRAM KILITLENDI"
                detail_text = "Ekran kilitleme komutu gönderildi."
                state_color = (80, 80, 255)

                # Gerçek kullanımda:
                # import ctypes
                # ctypes.windll.user32.LockWorkStation()
                write_log("Ekran kilitleme komutu tetiklendi.")

            elif detected_fingers == 2:
                light_on = not light_on
                system_state = "LIGHT_TOGGLE"
                status_text = "ODA ISIGI DEGISTIRILDI"
                detail_text = f"Oda ışığı: {'AÇIK' if light_on else 'KAPALI'} (simülasyon)"
                state_color = (60, 200, 120)

                write_log(f"Oda ışığı durumu değiştirildi: {light_on}")

            elif detected_fingers == 3:
                system_state = "SNAPSHOT"
                status_text = "EKRAM GORUNTUSU ALINDI"
                detail_text = "Klinik kayıt görüntüsü kaydedildi."
                state_color = (200, 160, 40)

                save_snapshot(frame, "klinik_kayit")

            elif detected_fingers == 4:
                system_state = "NURSE_CALL"
                status_text = "HEMŞIRE CAGRISI GONDERILDI"
                detail_text = "Hasta odasından hemşire çağrısı oluşturuldu."
                state_color = (40, 40, 230)

                if not nurse_call_sent:
                    write_log("HEMŞİRE ÇAĞRISI: Hasta odasından çağrı talebi oluşturuldu.")
                    speak("Hemşire çağrısı gönderildi.")
                    nurse_call_sent = True

        else:
            system_state = "VERIFYING"
            status_text = f"KOMUT DOGRULANIYOR: {detected_fingers} parmak"
            detail_text = f"Kalan süre: {remaining:.1f} saniye"
            state_color = (200, 180, 60)

    else:
        active_gesture = None
        gesture_start_time = None
        nurse_call_sent = False

        system_state = "READY"
        status_text = "SISTEM HAZIR"
        detail_text = "Bekleme modu"
        state_color = (180, 180, 180)

    # --------------------------- Dikkat izleme ---------------------------

    attention_text = "Yüz izleniyor..."

    if face_results.multi_face_landmarks:
        face_landmarks = face_results.multi_face_landmarks[0].landmark

        left_eye_indices = [33, 160, 158, 133, 153, 144]
        right_eye_indices = [362, 385, 387, 263, 373, 380]

        left_ear = calculate_ear(face_landmarks, left_eye_indices, frame_w, frame_h)
        right_ear = calculate_ear(face_landmarks, right_eye_indices, frame_w, frame_h)

        ear = (left_ear + right_ear) / 2.0
        pitch = estimate_head_pose(face_landmarks, frame_w, frame_h)

        # Göz kapanması
        if ear < EAR_CLOSED_THRESHOLD:
            if eye_closed_start is None:
                eye_closed_start = now

            eye_closed_duration = now - eye_closed_start

            if eye_closed_duration >= EYE_CLOSED_ALERT_SECONDS:
                alert_eye = True
        else:
            eye_closed_start = None
            alert_eye = False

        # Baş aşağı eğilme
        if pitch > HEAD_DOWN_PITCH_THRESHOLD:
            if head_down_start is None:
                head_down_start = now

            head_down_duration = now - head_down_start

            if head_down_duration >= HEAD_DOWN_ALERT_SECONDS:
                alert_head = True
        else:
            head_down_start = None
            alert_head = False

        if alert_eye and alert_head:
            attention_text = "UYARI: DIKKAT IZLEME SINYALI"
            write_log("UYARI: Uzun süre göz kapalı ve baş aşağı algılandı.")
        elif alert_eye:
            attention_text = "UYARI: UZUN SURE GOZ KAPALI"
            write_log("UYARI: Uzun süre göz kapalı algılandı.")
        elif alert_head:
            attention_text = "UYARI: BAS UZUN SURE ASAGIDA"
            write_log("UYARI: Baş uzun süre aşağıda algılandı.")
        else:
            attention_text = f"EAR: {ear:.2f} | Bas egimi: {pitch:.1f}"

        mp_draw.draw_landmarks(
            frame,
            face_results.multi_face_landmarks[0],
            mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_draw.DrawingSpec(
                color=(80, 160, 255),
                thickness=1,
                circle_radius=1,
            ),
        )

    else:
        eye_closed_start = None
        head_down_start = None
        alert_eye = False
        alert_head = False

        attention_text = "YUZ ALGILANAMADI"

    # --------------------------- Uyarı sesi ---------------------------

    if system_state != last_state:
        write_log(f"Sistem durumu: {system_state}")

        if system_state == "SCREEN_LOCK":
            speak("Ekran kilitlendi.")
        elif system_state == "LIGHT_TOGGLE":
            speak("Oda ışığı değiştirildi.")
        elif system_state == "SNAPSHOT":
            speak("Ekran görüntüsü alındı.")

        last_state = system_state

    if alert_eye and alert_head:
        speak("Dikkat izleme uyarısı.")

    # --------------------------- HUD ---------------------------

    draw_hud(
        frame,
        f"SAAT: {current_time} | FPS: {fps}",
        (20, 35),
    )

    draw_hud(
        frame,
        status_text,
        (20, 75),
        text_color=(255, 255, 255),
        background=state_color,
    )

    draw_hud(
        frame,
        detail_text,
        (20, 115),
    )

    draw_hud(
        frame,
        attention_text,
        (20, 155),
        text_color=(255, 255, 255),
        background=(40, 40, 40),
    )

    draw_hud(
        frame,
        f"Algilanan parmak: {detected_fingers} | Oda isigi: {'ACIK' if light_on else 'KAPALI'}",
        (20, 195),
    )

    cv2.imshow("Hastane Temassiz Kontrol ve Dikkat Izleme Paneli", frame)

    key = cv2.waitKey(5) & 0xFF

    if key == ord("q") or key == 27:
        write_log("Sistem kullanıcı tarafından kapatıldı.")
        speak("Sistem kapatılıyor.")
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
face_mesh.close()
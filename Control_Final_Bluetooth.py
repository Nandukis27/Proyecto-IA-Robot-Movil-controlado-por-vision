import cv2
import mediapipe as mp
import numpy as np
import pickle
import serial
import time
from collections import deque

# ============================
# CONFIGURACIÓN
# ============================
FACE_MODEL = "face_classifier.pkl"
GESTURE_MODEL = "gesture_classifier.pkl"
FACENET_MODEL = "openface.nn4.small2.v1.t7"

# Configuración Bluetooth
BLUETOOTH_PORT = "COM11"  # CAMBIAR según tu puerto
BAUD_RATE = 115200
BLUETOOTH_NAME = "ESP32_ROBOT"
ENABLE_BLUETOOTH = True  # Cambiar a False para modo prueba sin robot

# Umbrales de confianza
FACE_CONFIDENCE_THRESHOLD = 0.85
GESTURE_CONFIDENCE_THRESHOLD = 0.55 #
GESTURE_HOLD_TIME = 0.1     #

# Tiempo de autorización
AUTHORIZATION_TIMEOUT = 30

# Mapeo de gestos a comandos del ESP32
GESTURE_COMMANDS = {
    'puño_cerrado': 'S',      # Stop
    'pulgar_arriba': 'F',     # Forward
    'victoria': 'B',          # Back
    'señalar': 'L',           # Left
    'mano_abierta': 'R',      # Right
    'ok': 'T',                # Turbo
    'rock': 'D'               # Dance
}

COMMAND_NAMES = {
    'S': 'STOP',
    'F': 'ADELANTE',
    'B': 'ATRÁS',
    'L': 'IZQUIERDA',
    'R': 'DERECHA',
    'T': 'TURBO',
    'D': 'BAILE'
}

# CARGAR MODELOS
print("="*80)
print("SISTEMA DE CONTROL DE ROBOT - BLUETOOTH ESP32")
print("="*80)
print("\n[INFO] Cargando modelos de IA...\n")

# Cargar modelo facial
try:
    with open(FACE_MODEL, 'rb') as f:
        face_data = pickle.load(f)
    face_model = face_data["model"]
    face_le = face_data["label_encoder"]
    face_normalizer = face_data["normalizer"]
    authorized_users = face_data["names"]
    face_accuracy = face_data["accuracy"]
    
    print(f"[OK] Modelo facial cargado")
    print(f"     Precisión: {face_accuracy*100:.2f}%")
    print(f"     Usuarios autorizados: {authorized_users}\n")
    
except FileNotFoundError:
    print(f"[ERROR] No se encontró {FACE_MODEL}")
    exit()

# Cargar modelo de gestos
try:
    with open(GESTURE_MODEL, 'rb') as f:
        gesture_data = pickle.load(f)
    gesture_model = gesture_data["model"]
    gesture_scaler = gesture_data["scaler"]
    gesture_names = gesture_data["gestures"]
    gesture_accuracy = gesture_data["accuracy"]
    
    print(f"[OK] Modelo de gestos cargado")
    print(f"     Precisión: {gesture_accuracy*100:.2f}%")
    print(f"     Gestos disponibles: {len(gesture_names)}\n")
    
except FileNotFoundError:
    print(f"[ERROR] No se encontró {GESTURE_MODEL}")
    exit()

# Cargar FaceNet
try:
    facenet = cv2.dnn.readNetFromTorch(FACENET_MODEL)
    print(f"[OK] FaceNet cargado\n")
except:
    print(f"[ERROR] No se pudo cargar {FACENET_MODEL}\n")
    exit()

# CONECTAR BLUETOOTH
bt = None
if ENABLE_BLUETOOTH:
    print("="*80)
    print("📡 CONECTANDO CON ESP32 VÍA BLUETOOTH")
    print("="*80)
    print(f"\n[INFO] Intentando conectar a {BLUETOOTH_PORT}...")
    print(f"       Buscando dispositivo: {BLUETOOTH_NAME}\n")
    
    try:
        bt = serial.Serial(BLUETOOTH_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)  # Esperar estabilización
        print(f"[OK] Conectado exitosamente a {BLUETOOTH_PORT}")
        print(f"     Baudrate: {BAUD_RATE}")
        print(f"     Estado: {bt.is_open}\n")
        
        # Enviar comando de prueba
        bt.write(b'S\n')
        print("[TEST] Comando de prueba enviado: STOP\n")
        
    except serial.SerialException as e:
        print(f"[ERROR] No se pudo conectar al puerto {BLUETOOTH_PORT}")
        print(f"        Error: {e}\n")
        print("SOLUCIONES:")
        print("   1. Verifica que el ESP32 esté encendido y conectado")
        print(f"   2. Busca '{BLUETOOTH_NAME}' en dispositivos Bluetooth de tu sistema")
        print("   3. En Windows: Configuración > Bluetooth > Más opciones > Puertos COM")
        print("   4. Confirma el puerto COM asignado y actualiza BLUETOOTH_PORT")
        print("   5. Asegúrate de que no haya otra app usando el puerto\n")
        
        retry = input("¿Continuar en modo PRUEBA sin robot? (s/n): ")
        if retry.lower() != 's':
            exit()
        ENABLE_BLUETOOTH = False
        print("\n[INFO] Modo PRUEBA activado - No se enviarán comandos\n")
else:
    print("\n[INFO] Bluetooth DESHABILITADO - Modo prueba\n")

# ============================
# INICIALIZAR MEDIAPIPE
# ============================
mp_face_detection = mp.solutions.face_detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

face_detection = mp_face_detection.FaceDetection(
    model_selection=0, 
    min_detection_confidence=0.6
)

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ============================
# FUNCIONES
# ============================
def get_face_embedding(face_img):
    """Genera embedding facial de 128 dimensiones"""
    face_img = cv2.resize(face_img, (192, 192))
    blob = cv2.dnn.blobFromImage(face_img, 1.0/255, (96, 96), 
                                  (0, 0, 0), swapRB=True, crop=False)
    facenet.setInput(blob)
    embedding = facenet.forward()
    return embedding.flatten()

def extract_hand_landmarks(hand_landmarks):
    """Extrae landmarks de la mano y normaliza"""
    landmarks = []
    wrist = hand_landmarks.landmark[0]
    
    for landmark in hand_landmarks.landmark:
        landmarks.extend([
            landmark.x - wrist.x,
            landmark.y - wrist.y,
            landmark.z - wrist.z
        ])
    
    return np.array(landmarks)

def send_command_to_robot(command):
    """Envía comando al ESP32 vía Bluetooth"""
    if bt and bt.is_open and ENABLE_BLUETOOTH:
        try:
            bt.write(f"{command}\n".encode())
            return True
        except Exception as e:
            print(f"\n❌ Error enviando comando: {e}")
            return False
    return False

# CLASE DE AUTORIZACIÓN
class AuthorizationManager:
    def __init__(self):
        self.is_authorized = False
        self.current_user = None
        self.last_face_time = 0
        self.authorization_history = deque(maxlen=10)
        
    def update_face_detection(self, user_name, confidence):
        """Actualiza estado de autorización basado en detección facial"""
        self.authorization_history.append((user_name, confidence, time.time()))
        
        if confidence >= FACE_CONFIDENCE_THRESHOLD * 100:
            self.is_authorized = True
            self.current_user = user_name
            self.last_face_time = time.time()
            return True
        return False
    
    def check_authorization(self):
        """Verifica si la autorización sigue válida"""
        if self.is_authorized:
            elapsed = time.time() - self.last_face_time
            if elapsed > AUTHORIZATION_TIMEOUT:
                self.is_authorized = False
                self.current_user = None
                return False
        return self.is_authorized
    
    def get_time_remaining(self):
        """Obtiene tiempo restante de autorización"""
        if not self.is_authorized:
            return 0
        elapsed = time.time() - self.last_face_time
        remaining = AUTHORIZATION_TIMEOUT - elapsed
        return max(0, remaining)
    
    def reset(self):
        """Reinicia autorización"""
        self.is_authorized = False
        self.current_user = None
        self.last_face_time = 0
        self.authorization_history.clear()

# INTERFAZ VISUAL
def draw_connection_status(frame, bt_connected):
    """Dibuja estado de conexión Bluetooth"""
    h, w = frame.shape[:2]
    
    if ENABLE_BLUETOOTH and bt_connected:
        color = (0, 255, 0)
        status = "CONECTADO"
        icon_color = (0, 255, 0)
    elif ENABLE_BLUETOOTH:
        color = (0, 0, 255)
        status = "DESCONECTADO"
        icon_color = (0, 0, 255)
    else:
        color = (0, 165, 255)
        status = "MODO PRUEBA"
        icon_color = (0, 165, 255)
    
    # Panel de estado
    panel_w = 250
    panel_h = 60
    panel_x = w - panel_w - 20
    panel_y = 20
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), 
                 (panel_x + panel_w, panel_y + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
    
    cv2.rectangle(frame, (panel_x, panel_y), 
                 (panel_x + panel_w, panel_y + panel_h), color, 2)
    
    cv2.putText(frame, f"BT: {status}", (panel_x + 10, panel_y + 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    if ENABLE_BLUETOOTH:
        cv2.putText(frame, BLUETOOTH_PORT, (panel_x + 10, panel_y + 48), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    # Indicador visual (círculo)
    cv2.circle(frame, (panel_x + panel_w - 25, panel_y + 30), 10, icon_color, -1)
    
    return frame

def draw_authorization_panel(frame, auth_manager, face_detected, face_confidence):
    """Dibuja panel de autorización"""
    h, w = frame.shape[:2]
    
    panel_h = 140
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    
    y_offset = 35
    
    if auth_manager.is_authorized:
        status_text = f"AUTORIZADO: {auth_manager.current_user}"
        status_color = (0, 255, 0)
        
        cv2.putText(frame, status_text, (20, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)
        
        time_remaining = auth_manager.get_time_remaining()
        if not face_detected:
            time_text = f"Sin rostro - Expira en: {int(time_remaining)}s"
            time_color = (0, 165, 255) if time_remaining > 10 else (0, 0, 255)
        else:
            time_text = f"Autorizacion activa - Confianza: {face_confidence:.1f}%"
            time_color = (0, 255, 0)
        
        cv2.putText(frame, time_text, (20, y_offset + 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, time_color, 2)
        
        bar_x = 20
        bar_y = y_offset + 60
        bar_w = 400
        bar_h = 20
        
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
        
        progress = int((time_remaining / AUTHORIZATION_TIMEOUT) * bar_w)
        bar_color = (0, 255, 0) if time_remaining > 15 else (0, 165, 255) if time_remaining > 5 else (0, 0, 255)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + progress, bar_y + bar_h), bar_color, -1)
        
        cv2.putText(frame, "Control del robot HABILITADO", 
                    (20, y_offset + 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
    else:
        status_text = "ACCESO DENEGADO"
        status_color = (0, 0, 255)
        
        cv2.putText(frame, status_text, (20, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)
        
        if face_detected:
            if face_confidence < FACE_CONFIDENCE_THRESHOLD * 100:
                msg = f"Usuario no autorizado (Conf: {face_confidence:.1f}%)"
                msg_color = (0, 165, 255)
            else:
                msg = "Verificando identidad..."
                msg_color = (0, 255, 255)
        else:
            msg = "Muestra tu rostro para obtener acceso"
            msg_color = (200, 200, 200)
        
        cv2.putText(frame, msg, (20, y_offset + 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, msg_color, 2)
        
        cv2.putText(frame, f"Usuarios autorizados: {', '.join(authorized_users)}", 
                    (20, y_offset + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        cv2.putText(frame, "Control del robot BLOQUEADO", 
                    (20, y_offset + 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    return frame

def draw_gesture_panel(frame, gesture_data, is_authorized, last_sent_command, command_sent_success):
    """Dibuja panel de gestos con estado de envío"""
    h, w = frame.shape[:2]
    
    panel_w = 450
    panel_h = 280
    panel_x = w - panel_w - 20
    panel_y = 160
    
    overlay = frame.copy()
    
    if is_authorized and gesture_data["detected"]:
        cv2.rectangle(overlay, (panel_x, panel_y), 
                     (panel_x + panel_w, panel_y + panel_h), (0, 100, 0), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        
        cv2.rectangle(frame, (panel_x, panel_y), 
                     (panel_x + panel_w, panel_y + panel_h), (0, 255, 0), 2)
        
        cv2.putText(frame, "COMANDO DETECTADO", 
                    (panel_x + 10, panel_y + 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(frame, f"Gesto: {gesture_data['gesture']}", 
                    (panel_x + 10, panel_y + 75), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        cmd = GESTURE_COMMANDS.get(gesture_data['gesture'], '?')
        cmd_name = COMMAND_NAMES.get(cmd, 'DESCONOCIDO')
        cv2.putText(frame, f"Comando: {cmd} - {cmd_name}", 
                    (panel_x + 10, panel_y + 115), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.putText(frame, f"Confianza: {gesture_data['confidence']:.1f}%", 
                    (panel_x + 10, panel_y + 155), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Estado de envío
        if last_sent_command == cmd:
            if ENABLE_BLUETOOTH:
                if command_sent_success:
                    send_status = "ENVIADO AL ROBOT"
                    send_color = (0, 255, 0)
                else:
                    send_status = "ERROR AL ENVIAR"
                    send_color = (0, 0, 255)
            else:
                send_status = "MODO PRUEBA (No enviado)"
                send_color = (0, 165, 255)
            
            cv2.putText(frame, send_status, 
                        (panel_x + 10, panel_y + 195), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, send_color, 2)
        
        cv2.circle(frame, (panel_x + panel_w - 40, panel_y + 40), 
                   25, (0, 255, 0), -1)
        cv2.putText(frame, "OK", (panel_x + panel_w - 55, panel_y + 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
    elif not is_authorized:
        cv2.rectangle(overlay, (panel_x, panel_y), 
                     (panel_x + panel_w, panel_y + panel_h), (0, 0, 100), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        
        cv2.rectangle(frame, (panel_x, panel_y), 
                     (panel_x + panel_w, panel_y + panel_h), (0, 0, 255), 2)
        
        cv2.putText(frame, "CONTROL BLOQUEADO", 
                    (panel_x + 10, panel_y + 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.putText(frame, "Necesitas autorizacion", 
                    (panel_x + 10, panel_y + 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        cv2.putText(frame, "para controlar el robot", 
                    (panel_x + 10, panel_y + 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
    else:
        cv2.rectangle(overlay, (panel_x, panel_y), 
                     (panel_x + panel_w, panel_y + panel_h), (50, 50, 50), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        
        cv2.rectangle(frame, (panel_x, panel_y), 
                     (panel_x + panel_w, panel_y + panel_h), (150, 150, 150), 2)
        
        cv2.putText(frame, "Esperando gesto...", 
                    (panel_x + 10, panel_y + 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        cv2.putText(frame, "Haz un gesto con tu mano", 
                    (panel_x + 10, panel_y + 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
    
    return frame

def draw_statistics(frame, stats):
    """Dibuja estadísticas en la parte inferior"""
    h, w = frame.shape[:2]
    
    stats_y = h - 100
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, stats_y), (w, h - 40), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    col1 = 20
    col2 = w // 3
    col3 = 2 * w // 3
    
    cv2.putText(frame, f"FPS: {stats['fps']:.1f}", 
                (col1, stats_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(frame, f"Detecciones faciales: {stats['face_count']}", 
                (col1, stats_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    cv2.putText(frame, f"Autorizaciones: {stats['auth_count']}", 
                (col2, stats_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(frame, f"Gestos detectados: {stats['gesture_count']}", 
                (col2, stats_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    cv2.putText(frame, f"Comandos enviados: {stats['command_count']}", 
                (col3, stats_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(frame, f"Errores: {stats['error_count']}", 
                (col3, stats_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # Instrucciones
    cv2.rectangle(frame, (0, h - 40), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, "ESC=Salir | R=Reset Auth | S=Stop Robot | ESPACIO=Estadisticas", 
                (20, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    
    return frame

# ============================
# SISTEMA PRINCIPAL
# ============================
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

auth_manager = AuthorizationManager()
gesture_buffer = deque(maxlen=int(GESTURE_HOLD_TIME * 30))

stats = {
    'fps': 0,
    'face_count': 0,
    'auth_count': 0,
    'gesture_count': 0,
    'command_count': 0,
    'error_count': 0
}

last_frame_time = time.time()
last_sent_command = None
command_sent_success = False

print("="*80)
print("SISTEMA DE CONTROL INICIADO")
print("="*80)
print("\nInstrucciones:")
print("   1. Muestra tu rostro para autorizarte")
print("   2. Haz gestos para controlar el robot")
if ENABLE_BLUETOOTH:
    print("   3. Los comandos se envían automáticamente al ESP32")
else:
    print("   3. MODO PRUEBA - Los comandos solo se mostrarán")
print("\nControles:")
print("   ESC     → Salir (envía STOP al robot)")
print("   R       → Restablecer autorización")
print("   S       → Enviar STOP de emergencia")
print("   ESPACIO → Ver estadísticas")
print("="*80 + "\n")

# LOOP PRINCIPAL
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        current_time = time.time()
        fps = 1 / (current_time - last_frame_time) if (current_time - last_frame_time) > 0 else 0
        last_frame_time = current_time
        stats['fps'] = fps
        
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        face_detected = False
        face_confidence = 0
        gesture_data = {"detected": False, "gesture": "", "confidence": 0}
        
        # ============================
        # RECONOCIMIENTO FACIAL
        # ============================
        face_results = face_detection.process(frame_rgb)
        
        if face_results.detections:
            for detection in face_results.detections:
                mp_drawing.draw_detection(frame, detection)
                
                bbox = detection.location_data.relative_bounding_box
                x_min = int(bbox.xmin * w)
                y_min = int(bbox.ymin * h)
                box_w = int(bbox.width * w)
                box_h = int(bbox.height * h)
                
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                
                face_img = frame[y_min:y_min+box_h, x_min:x_min+box_w]
                
                if face_img.size > 0:
                    embedding = get_face_embedding(face_img)
                    embedding = face_normalizer.transform([embedding])
                    
                    proba = face_model.predict_proba(embedding)[0]
                    confidence = max(proba)
                    face_confidence = confidence * 100
                    
                    stats['face_count'] += 1
                    face_detected = True
                    
                    if confidence >= FACE_CONFIDENCE_THRESHOLD:
                        label_idx = face_model.predict(embedding)[0]
                        user_name = authorized_users[face_le.inverse_transform([label_idx])[0]]
                        
                        was_authorized = auth_manager.is_authorized
                        auth_manager.update_face_detection(user_name, face_confidence)
                        
                        if not was_authorized and auth_manager.is_authorized:
                            stats['auth_count'] += 1
                            print(f"\nAUTORIZADO: {user_name} (Confianza: {face_confidence:.1f}%)")
        
        auth_manager.check_authorization()
        
        # ============================
        # DETECCIÓN DE GESTOS
        # ============================
        if auth_manager.is_authorized:
            hand_results = hands.process(frame_rgb)
            
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
                    
                    landmarks = extract_hand_landmarks(hand_landmarks)
                    landmarks = gesture_scaler.transform([landmarks])
                    
                    gesture_pred = gesture_model.predict(landmarks)[0]
                    proba = gesture_model.predict_proba(landmarks)[0]
                    confidence = max(proba) * 100
                    
                    gesture_buffer.append((gesture_pred, confidence))
                    
                    if len(gesture_buffer) == gesture_buffer.maxlen:
                        gestures = [g[0] for g in gesture_buffer]
                        confidences = [g[1] for g in gesture_buffer]
                        
                        most_common = max(set(gestures), key=gestures.count)
                        avg_confidence = np.mean(confidences)
                        
                        if (gestures.count(most_common) >= len(gestures) * 0.7 and 
                            avg_confidence >= GESTURE_CONFIDENCE_THRESHOLD * 100):
                            
                            gesture_data = {
                                "detected": True,
                                "gesture": most_common,
                                "confidence": avg_confidence
                            }
                            
                            stats['gesture_count'] += 1
                            
                            if most_common in GESTURE_COMMANDS:
                                command = GESTURE_COMMANDS[most_common]
                                cmd_name = COMMAND_NAMES.get(command, 'DESCONOCIDO')
                                
                                if command != last_sent_command:
                                    # ENVIAR COMANDO AL ROBOT
                                    if send_command_to_robot(command):
                                        stats['command_count'] += 1
                                        command_sent_success = True
                                        print(f"🤖 ROBOT: {most_common:20s} → {command} ({cmd_name:12s}) | Usuario: {auth_manager.current_user}")
                                    else:
                                        stats['error_count'] += 1
                                        command_sent_success = False
                                        if ENABLE_BLUETOOTH:
                                            print(f"❌ ERROR: No se pudo enviar comando {command}")
                                    
                                    last_sent_command = command
                                    gesture_buffer.clear()
        
        # ============================
        # DIBUJAR INTERFAZ
        # ============================
        bt_connected = (bt and bt.is_open) if ENABLE_BLUETOOTH else False
        
        frame = draw_connection_status(frame, bt_connected)
        frame = draw_authorization_panel(frame, auth_manager, face_detected, face_confidence)
        frame = draw_gesture_panel(frame, gesture_data, auth_manager.is_authorized, 
                                   last_sent_command, command_sent_success)
        frame = draw_statistics(frame, stats)
        
        cv2.imshow("Control de Robot - Sistema Completo", frame)
        
        # ============================
        # CONTROLES
        # ============================
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC
            print("\n[INFO] Cerrando sistema...")
            if send_command_to_robot('S'):
                print("[OK] Comando STOP enviado al robot")
            break
        
        elif key == ord('r') or key == ord('R'):
            auth_manager.reset()
            if send_command_to_robot('S'):
                print("\n[INFO] Autorización restablecida y robot detenido")
        
        elif key == ord('s') or key == ord('S'):
            if send_command_to_robot('S'):
                print("\n[EMERGENCY] STOP de emergencia enviado")
        
        elif key == 32:  # ESPACIO
            print("\n" + "="*80)
            print("ESTADÍSTICAS DEL SISTEMA")
            print("="*80)
            print(f"FPS promedio: {stats['fps']:.1f}")
            print(f"Detecciones faciales: {stats['face_count']}")
            print(f"Autorizaciones: {stats['auth_count']}")
            print(f"Gestos detectados: {stats['gesture_count']}")
            print(f"Comandos enviados: {stats['command_count']}")
            print(f"Errores de comunicación: {stats['error_count']}")
            if auth_manager.is_authorized:
                print(f"Estado: AUTORIZADO ({auth_manager.current_user})")
                print(f"Tiempo restante: {auth_manager.get_time_remaining():.1f}s")
            else:
                print(f"Estado: NO AUTORIZADO")
            if ENABLE_BLUETOOTH:
                print(f"Bluetooth: {'CONECTADO' if bt_connected else 'DESCONECTADO'}")
            else:
                print(f"Bluetooth: MODO PRUEBA")
            print("="*80 + "\n")

except KeyboardInterrupt:
    print("\n[INFO] Interrumpido por usuario")
    if send_command_to_robot('S'):
        print("[OK] Robot detenido")

# ============================
# LIMPIEZA
# ============================
if bt and bt.is_open:
    bt.close()
    print("[OK] Bluetooth desconectado")

cap.release()
cv2.destroyAllWindows()

print("\n" + "="*80)
print("RESUMEN FINAL")
print("="*80)
print(f"Detecciones faciales: {stats['face_count']}")
print(f"Autorizaciones: {stats['auth_count']}")
print(f"Gestos detectados: {stats['gesture_count']}")
print(f"Comandos enviados: {stats['command_count']}")
print(f"Errores: {stats['error_count']}")
print("="*80)
print("Sistema cerrado correctamente")
print("="*80 + "\n")
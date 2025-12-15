import cv2
import mediapipe as mp
import os
import time

NOMBRE_USUARIO = "fernando_2"   # Cambia tu nombre aqui
CARPETA_SALIDA = f"dataset_Usuario/{NOMBRE_USUARIO}"
NUMERO_IMAGENES = 200         # Cantidad recomendada de imagenes a capturar

# Carpeta del dataset
os.makedirs(CARPETA_SALIDA, exist_ok=True)

# Mediapipe 
mp_deteccion_rostro = mp.solutions.face_detection # Modulo de deteccion de rostro
mp_dibujo = mp.solutions.drawing_utils 

detector_rostro = mp_deteccion_rostro.FaceDetection(
    model_selection=0, # 0 modelo lijero (cerca, <2m) , 1 modelo pesado (lejos, >2m)
    min_detection_confidence=0.6
)

def recortar_rostro(imagen, caja_delimitadora_relativa):
    # Obtener dimensiones de la imagen
    alto, ancho, _ = imagen.shape
    # Extraer y convertir coordenadas relativas a absolutas
    x_min = int(caja_delimitadora_relativa.xmin * ancho)
    y_min = int(caja_delimitadora_relativa.ymin * alto)
    ancho_caja = int(caja_delimitadora_relativa.width * ancho)
    alto_caja = int(caja_delimitadora_relativa.height * alto)
    # Asegurar que las coordenadas no sean negativas
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    # Calcular las coordenadas de fin para el recorte
    x_max = x_min + ancho_caja
    y_max = y_min + alto_caja
    # Recortar la region del rostro
    rostro_recortado = imagen[y_min:y_max, x_min:x_max]
    return rostro_recortado

def mostrar_estado_captura(frame, esta_capturando, contador_actual, max_imagenes):
    if esta_capturando:
        texto = f"CAPTURANDO: {contador_actual}/{max_imagenes}"
        color = (0, 255, 0) # Verde
    else:
        texto = "Presiona 'G' para iniciar"
        color = (0, 165, 255) # Naranja/Amarillo
    cv2.putText(frame, texto, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

def iniciar_captura_dataset():
    cap = cv2.VideoCapture(1)
    contador_imagenes = 0
    esta_capturando = False # Estado de captura

    print("\n[INFO] Camara iniciada")
    print("Presiona 'G' para comenzar la captura")
    print("Presiona 'ESC' para salir\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] No se pudo leer el frame de la camara.")
            break

        # Procesar el frame para deteccion de rostros
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
        resultados = detector_rostro.process(frame_rgb)
        
        # Dibujar el estado en el frame
        mostrar_estado_captura(frame, esta_capturando, contador_imagenes, NUMERO_IMAGENES)

        if resultados.detections:
            for deteccion in resultados.detections:
                caja_delimitadora = deteccion.location_data.relative_bounding_box
                
                # Dibujar la caja delimitadora en el frame
                mp_dibujo.draw_detection(frame, deteccion)

                if esta_capturando:
                    # 1. Recortar rostro
                    rostro_img = recortar_rostro(frame, caja_delimitadora)

                    # Verificar si el recorte fue exitoso
                    if rostro_img.size == 0 or rostro_img is None:
                        continue

                    # 2. Normalizar tamano para el entrenamiento (192x192)
                    rostro_img = cv2.resize(rostro_img, (192, 192))

                    # 3. Guardar imagen
                    ruta_img = f"{CARPETA_SALIDA}/{NOMBRE_USUARIO}_{contador_imagenes}.jpg"
                    cv2.imwrite(ruta_img, rostro_img)
                    contador_imagenes += 1
                    
                    # Salir del bucle si se alcanza el numero de imagenes
                    if contador_imagenes >= NUMERO_IMAGENES:
                        esta_capturando = False # Detener la captura automaticamente
                        break
        cv2.imshow("Capturando Dataset Facial", frame)

        # Manejo de teclas
        key = cv2.waitKey(1) & 0xFF
        1
        if key == ord('g') or key == ord('G'): # Tecla G
            if not esta_capturando:
                esta_capturando = True
                print("[INFO] Captura iniciada. Mueve tu cabeza y cambia expresiones.")
        elif key == 27: # ESC para salir
            break
        
        # Salir si se alcanzo el limite y el usuario no detuvo la captura antes
        if contador_imagenes >= NUMERO_IMAGENES and not esta_capturando:
             break # Salir del bucle principal

    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()

    print(f"\n[OK] Dataset capturado con exito en: {CARPETA_SALIDA}")
    print(f"Total de imagenes: {contador_imagenes}")

# Ejecutar la funcion principal
if __name__ == "__main__":
    iniciar_captura_dataset()
import cv2
import mediapipe as mp
import numpy as np
import os
import json
import time
from typing import Dict, Any, List

# ============================
# ⚙️ CONFIGURACION
# ============================

# Mapeo de gestos (clave: nombre)
GESTOS: Dict[str, str] = {
    '0': 'puno_cerrado',
    '1': 'pulgar_arriba',
    '2': 'victoria',
    '3': 'ok',
    '4': 'mano_abierta',
    '5': 'senalar',
    '6': 'rock' 
}

CARPETA_SALIDA = "dataset_manos"
MUESTRAS_POR_GESTO = 200  # Muestras por gesto
MUESTRAS_POR_POSICION = 100  # Muestras antes de pausar
DEMORA_CAPTURA = 0.1  # Segundos entre capturas
PAUSA_AUTOMATICA_HABILITADA = True  # Pausar automaticamente cada N muestras

# Crear carpeta principal
os.makedirs(CARPETA_SALIDA, exist_ok=True)

# 
# INICIALIZAR MEDIAPIPE
mp_manos = mp.solutions.hands
mp_dibujo = mp.solutions.drawing_utils
mp_estilos_dibujo = mp.solutions.drawing_styles

detector_manos = mp_manos.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

def extraer_puntos_clave(puntos_clave_mano):
    """
    Extrae las coordenadas (x, y, z) de los 21 puntos de la mano
    y las normaliza respecto a la muneca (punto 0)
    """
    puntos_clave = []
    
    # Obtener coordenadas de la muneca (punto 0) como referencia
    muneca = puntos_clave_mano.landmark[0]
    
    for punto in puntos_clave_mano.landmark:
        # Normalizar respecto a la muneca
        puntos_clave.extend([
            punto.x - muneca.x,
            punto.y - muneca.y,
            punto.z - muneca.z
        ])
    
    return puntos_clave

def dibujar_interfaz(frame, nombre_gesto, contador_actual, total_muestras, esta_pausado, esta_capturando, contador_posicion):
    alto, ancho = frame.shape[:2]
    
    # Constantes
    COLOR_TITULO = (0, 255, 255) # Amarillo/Cyan
    COLOR_CAPTURANDO = (0, 255, 0) # Verde
    COLOR_PAUSA = (0, 165, 255) # Naranja
    COLOR_TEXTO = (200, 200, 200)

    # ========== PANEL SUPERIOR ==========
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (ancho, 180), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    
    # Título y gesto actual
    cv2.putText(frame, f"GESTO: {nombre_gesto.upper()}", 
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_TITULO, 2)
    
    # Contador principal
    cv2.putText(frame, f"Progreso: {contador_actual}/{total_muestras}", 
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_CAPTURANDO, 2)
    
    # Contador de posicion
    restante_posicion = MUESTRAS_POR_POSICION - contador_posicion
    if restante_posicion > 0 and esta_capturando:
        color_pos = COLOR_PAUSA if restante_posicion <= 5 else COLOR_TEXTO
        cv2.putText(frame, f"En esta posicion: {contador_posicion}/{MUESTRAS_POR_POSICION}", 
                    (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_pos, 2)
    
    # Estado
    if esta_pausado:
        cv2.putText(frame, "PAUSADO - Cambia de posicion", 
                    (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_PAUSA, 2)
    elif esta_capturando:
        cv2.putText(frame, "CAPTURANDO...", 
                    (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_CAPTURANDO, 2)
    else:
        cv2.putText(frame, "Presiona ESPACIO para comenzar", 
                    (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXTO, 2)
    
    # ========== BARRA DE PROGRESO ==========
    bar_x, bar_y = 20, alto - 80
    bar_w, bar_h = ancho - 40, 30
    
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
    
    progreso_ancho = int((contador_actual / total_muestras) * bar_w)
    color_barra = COLOR_CAPTURANDO if contador_actual < total_muestras else COLOR_TITULO
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + progreso_ancho, bar_y + bar_h), color_barra, -1)
    
    porcentaje = (contador_actual / total_muestras) * 100
    texto_porcentaje = f"{porcentaje:.0f}%"
    text_size = cv2.getTextSize(texto_porcentaje, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    text_x = bar_x + (bar_w - text_size[0]) // 2
    text_y = bar_y + (bar_h + text_size[1]) // 2
    cv2.putText(frame, texto_porcentaje, (text_x, text_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # ========== PANEL DE INSTRUCCIONES ==========
    instrucciones_y = alto - 45
    cv2.rectangle(frame, (0, instrucciones_y), (ancho, alto), (0, 0, 0), -1)
    
    cv2.putText(frame, "ESPACIO=Iniciar/Reanudar | P=Pausar | R=Reiniciar gesto | ESC=Salir", 
                (20, instrucciones_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXTO, 1)
    
    # ========== INDICADORES VISUALES ==========
    if esta_capturando and not esta_pausado:
        pulso = int(abs(np.sin(time.time() * 5)) * 20) + 10
        cv2.circle(frame, (ancho - 50, 50), pulso, COLOR_CAPTURANDO, -1)
    
    if esta_pausado:
        cv2.rectangle(frame, (ancho - 70, 30), (ancho - 50, 70), COLOR_PAUSA, -1)
        cv2.rectangle(frame, (ancho - 45, 30), (ancho - 25, 70), COLOR_PAUSA, -1)
    
    return frame

def mostrar_pantalla_consejos(frame, nombre_gesto):
    alto, ancho = frame.shape[:2]
    
    # Fondo semi-transparente
    overlay = frame.copy()
    cv2.rectangle(overlay, (50, 50), (ancho-50, alto-50), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)
    
    # Borde
    cv2.rectangle(frame, (50, 50), (ancho-50, alto-50), (0, 255, 255), 3)
    
    y_offset = 100
    
    # Título
    cv2.putText(frame, f"Preparate para capturar: {nombre_gesto.upper()}", 
                (80, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    
    y_offset += 60
    
    # Consejos
    num_posiciones = int(np.ceil(MUESTRAS_POR_GESTO/MUESTRAS_POR_POSICION))
    
    consejos = [
        "Vas a capturar en MULTIPLES POSICIONES",
        "",
        f"• El sistema pausara cada {MUESTRAS_POR_POSICION} capturas ({num_posiciones} posiciones totales)",
        "• Usa la pausa para cambiar:",
        "  - Posicion de la mano (centro, izquierda, derecha, arriba, abajo)",
        "  - Distancia a la camara (cerca, media, lejos)",
        "  - Angulo de la mano (frontal, ladeada)",
        "  - Rotacion ligera del gesto",
        "",
        "• Tambien puedes pausar manualmente con 'P'",
        "• Haz el gesto de forma clara y natural",
        "",
        "Presiona ESPACIO para comenzar..."
    ]
    
    for consejo in consejos:
        cv2.putText(frame, consejo, (80, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        y_offset += 35
    
    return frame

def mostrar_superposicion_pausa(frame, contador_actual, total_muestras, numero_posicion):
    """
    Muestra overlay cuando está en pausa
    """
    alto, ancho = frame.shape[:2]
    
    # Overlay semi-transparente
    overlay = frame.copy()
    cv2.rectangle(overlay, (ancho//4, alto//4), (3*ancho//4, 3*alto//4), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    
    # Borde
    cv2.rectangle(frame, (ancho//4, alto//4), (3*ancho//4, 3*alto//4), (0, 165, 255), 3)
    
    y_center = alto//2
    
    # Título
    cv2.putText(frame, "PAUSA", 
                (ancho//2 - 70, y_center - 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 3)
    
    # Instrucciones
    num_posiciones_totales = int(np.ceil(total_muestras/MUESTRAS_POR_POSICION))
    texts = [
        "Cambia a una nueva posicion:",
        "",
        "• Mueve la mano a otro lugar",
        "• Varia la distancia",
        "• Cambia el angulo",
        "",
        f"Posicion {numero_posicion} de {num_posiciones_totales}",
        "",
        "Presiona ESPACIO para continuar"
    ]
    
    y_offset = y_center - 20
    for text in texts:
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        text_x = ancho//2 - text_size[0]//2
        cv2.putText(frame, text, (text_x, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        y_offset += 35
    
    return frame

def mostrar_referencia_gestos():
    """
    Muestra una referencia visual de los gestos en consola
    """
    print("\n" + "="*80)
    print("📋 GESTOS A CAPTURAR:")
    print("="*80)
    for key, gesto in GESTOS.items():
        print(f"  {key}. {gesto}")
    print("="*80)
    print("\n💡 ESTRATEGIA DE CAPTURA CON PAUSAS:")
    print(f"  • Se capturan {MUESTRAS_POR_GESTO} muestras por gesto")
    print(f"  • Pausa automatica cada {MUESTRAS_POR_POSICION} capturas")
    print(f"  • Esto te da {int(np.ceil(MUESTRAS_POR_GESTO/MUESTRAS_POR_POSICION))} posiciones diferentes")
    print("\n  En cada posicion:")
    print("    1. Coloca tu mano en un lugar diferente")
    print("    2. Varia distancia y angulo")
    print("    3. Presiona ESPACIO para continuar capturando")
    print("\n⌨️  CONTROLES:")
    print("  ESPACIO → Iniciar/Reanudar captura")
    print("  P → Pausar manualmente")
    print("  R → Reiniciar gesto actual")
    print("  ESC → Salir")
    print("="*80 + "\n")

# ============================
# 🎬 FUNCION PRINCIPAL DE CAPTURA
# ============================

def iniciar_captura_gestos():
    cap = cv2.VideoCapture(1) # Cambiado a 0 para mayor compatibilidad con notebooks/entornos locales

    # Intentar aumentar resolucion
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    mostrar_referencia_gestos()

    # Datos recolectados
    dataset: Dict[str, Any] = {
        'puntos_clave': [],
        'etiquetas': []
    }

    # Variables de control
    indice_gesto_actual = 0
    claves_gestos = list(GESTOS.keys())
    
    esta_capturando = False
    esta_pausado = False
    contador_muestras = 0
    contador_posicion = 0  
    numero_posicion = 1  
    tiempo_ultima_captura = 0
    mostrar_consejos = True

    print("[INFO] Camara iniciada.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if indice_gesto_actual >= len(claves_gestos):
            break
        
        frame = cv2.flip(frame, 1) # Voltear horizontalmente
        alto, ancho = frame.shape[:2]
        
        clave_gesto_actual = claves_gestos[indice_gesto_actual]
        nombre_gesto_actual = GESTOS[clave_gesto_actual]
        
        # --- Pantalla de Consejos ---
        if mostrar_consejos:
            frame = mostrar_pantalla_consejos(frame, nombre_gesto_actual)
            cv2.imshow("Captura de Gestos con Pausas", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 32:  # ESPACIO
                mostrar_consejos = False
                print(f"\n[INFO] 🎬 Iniciando captura de '{nombre_gesto_actual}'...")
                print(f"       Posicion 1 de {int(np.ceil(MUESTRAS_POR_GESTO/MUESTRAS_POR_POSICION))}")
            elif key == 27:  # ESC
                print("\n[INFO] Captura cancelada")
                break
            continue
        
        # --- Procesamiento de MediaPipe ---
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultados = detector_manos.process(frame_rgb)
        
        mano_detectada = False
        if resultados.multi_hand_landmarks:
            for puntos_clave_mano in resultados.multi_hand_landmarks:
                mano_detectada = True
                
                # Dibujar landmarks
                mp_dibujo.draw_landmarks(
                    frame, puntos_clave_mano, mp_manos.HAND_CONNECTIONS,
                    mp_estilos_dibujo.get_default_hand_landmarks_style(),
                    mp_estilos_dibujo.get_default_hand_connections_style()
                )
                
                # --- Lógica de Captura ---
                if esta_capturando and not esta_pausado:
                    tiempo_actual = time.time()
                    
                    if tiempo_actual - tiempo_ultima_captura >= DEMORA_CAPTURA:
                        # Extraer puntos clave normalizados
                        puntos_clave = extraer_puntos_clave(puntos_clave_mano)
                        
                        # Guardar datos
                        dataset['puntos_clave'].append(puntos_clave)
                        dataset['etiquetas'].append(nombre_gesto_actual)
                        
                        contador_muestras += 1
                        contador_posicion += 1
                        tiempo_ultima_captura = tiempo_actual
                        
                        # Feedback visual
                        cv2.circle(frame, (ancho-50, 50), 25, (0, 255, 0), -1)
                        
                        # Pausa automatica
                        if PAUSA_AUTOMATICA_HABILITADA and contador_posicion >= MUESTRAS_POR_POSICION and contador_muestras < MUESTRAS_POR_GESTO:
                            esta_pausado = True
                            contador_posicion = 0
                            numero_posicion += 1
                            print(f"\n[PAUSA] 🔄 Cambia a una nueva posicion")
                            print(f"        Posicion {numero_posicion} de {int(np.ceil(MUESTRAS_POR_GESTO/MUESTRAS_POR_POSICION))}")
                            print(f"        Progreso: {contador_muestras}/{MUESTRAS_POR_GESTO}")
                        
                        # Gesto completado
                        if contador_muestras >= MUESTRAS_POR_GESTO:
                            print(f"\n[OK] ✅ Gesto '{nombre_gesto_actual}' completado!")
                            
                            indice_gesto_actual += 1
                            contador_muestras = 0
                            contador_posicion = 0
                            numero_posicion = 1
                            esta_capturando = False
                            esta_pausado = False
                            mostrar_consejos = True
                            
                            if indice_gesto_actual < len(claves_gestos):
                                print(f"\n[INFO] Siguiente gesto: {GESTOS[claves_gestos[indice_gesto_actual]]}")
        
        if not mano_detectada and esta_capturando and not esta_pausado:
            # Advertencia si no se detecta mano
            cv2.putText(frame, "⚠️ MANO NO DETECTADA", 
                       (ancho//2 - 200, alto//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        
        # Dibujar interfaz principal
        frame = dibujar_interfaz(frame, nombre_gesto_actual, contador_muestras, MUESTRAS_POR_GESTO, 
                                 esta_pausado, esta_capturando, contador_posicion)
        
        # Overlay de pausa
        if esta_pausado:
            frame = mostrar_superposicion_pausa(frame, contador_muestras, MUESTRAS_POR_GESTO, numero_posicion)
        
        # Mostrar frame
        cv2.imshow("Captura de Gestos con Pausas", frame)
        
        # --- Controles de teclado ---
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC
            print("\n[INFO] Captura cancelada por el usuario")
            break
            
        elif key == 32:  # ESPACIO
            if esta_pausado:
                esta_pausado = False
                print(f"[INFO] ▶️  Reanudando captura...")
            elif not esta_capturando:
                esta_capturando = True
                esta_pausado = False
                print(f"[INFO] 🎬 Capturando gesto '{nombre_gesto_actual}'...")
                
        elif key == ord('p') or key == ord('P'):  # P para pausar manualmente
            if esta_capturando and not esta_pausado:
                esta_pausado = True
                print(f"\n[PAUSA] ⏸️  Pausado manualmente")
                print(f"        Progreso: {contador_muestras}/{MUESTRAS_POR_GESTO}")
                
        elif key == ord('r') or key == ord('R'):  # R para reiniciar gesto
            # Eliminar muestras del gesto actual del dataset
            indices_a_mantener = [i for i, label in enumerate(dataset['etiquetas']) 
                                   if label != nombre_gesto_actual]
            dataset['puntos_clave'] = [dataset['puntos_clave'][i] for i in indices_a_mantener]
            dataset['etiquetas'] = [dataset['etiquetas'][i] for i in indices_a_mantener]

            contador_muestras = 0
            contador_posicion = 0
            numero_posicion = 1
            esta_capturando = False
            esta_pausado = False
            print(f"\n[INFO] 🔄 Gesto '{nombre_gesto_actual}' reiniciado")

    # ============================
    # GUARDAR DATASET
    # ============================
    cap.release()
    cv2.destroyAllWindows()

    if len(dataset['puntos_clave']) > 0:
        # Convertir a numpy arrays
        # Renombrar 'puntos_clave' a 'landmarks' para compatibilidad con el siguiente script
        puntos_clave_np = np.array(dataset['puntos_clave'])
        etiquetas_np = np.array(dataset['etiquetas'])
        
        # Guardar como archivo numpy (utilizando 'landmarks' y 'labels' como claves internas)
        np.savez(f"{CARPETA_SALIDA}/gestos_manos_dataset.npz",
                 landmarks=puntos_clave_np,
                 labels=etiquetas_np)
        
        # Guardar información del dataset
        info = {
            'total_muestras': len(puntos_clave_np),
            'gestos': GESTOS,
            'muestras_por_gesto': MUESTRAS_POR_GESTO,
            'muestras_por_posicion': MUESTRAS_POR_POSICION,
            'num_caracteristicas': puntos_clave_np.shape[1],
            'estrategia_captura': 'multi-posicion con pausas'
        }
        
        with open(f"{CARPETA_SALIDA}/dataset_info.json", 'w') as f:
            json.dump(info, f, indent=4)
        
        # Resumen
        print("\n" + "="*80)
        print("📊 RESUMEN DEL DATASET")
        print("="*80)
        print(f"✅ Total de muestras: {len(puntos_clave_np)}")
        print(f"📋 Gestos capturados: {len(set(etiquetas_np))}")
        print(f"📍 Caracteristicas por muestra: {puntos_clave_np.shape[1]} (21 puntos x 3 coord.)")
        print(f"💾 Archivo guardado: {CARPETA_SALIDA}/gestos_manos_dataset.npz")
        
        # Distribución por gesto
        print("\n📊 Distribucion de muestras:")
        unique, counts = np.unique(etiquetas_np, return_counts=True)
        for gesture, count in zip(unique, counts):
            print(f"   {gesture:20s}: {count:3d} muestras")
        
        print("="*80)
        print("\n🚀 Siguiente paso: Entrenar el clasificador con este dataset.")
    else:
        print("\n[ERROR] No se capturo ningun dato. Ejecucion finalizada.")


# Ejecutar la funcion principal
if __name__ == "__main__":
    iniciar_captura_gestos()
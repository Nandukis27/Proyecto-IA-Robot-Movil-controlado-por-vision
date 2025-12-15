import cv2
import numpy as np
import os
import pickle
from pathlib import Path
from typing import Dict, Any, List


CARPETA_DATASET = "dataset_Usuario"  # Carpeta donde esta el dataset de usuarios
ARCHIVO_SALIDA = "embeddings_faciales.pkl"  # Archivo que tienen los embeddings

# Ruta del modelo FaceNet (Caffe)
FACENET_MODELO = "openface.nn4.small2.v1.t7" 



def cargar_modelo(ruta_modelo: str):
    print(f"[INFO] Intentando cargar modelo FaceNet desde: {ruta_modelo}...")
    try:
        # cv2.dnn.readNetFromTorch se usa porque FaceNet es un modelo Torch (PyTorch)
        # aunque el archivo termine en .t7 (Torch7)
        red_neural = cv2.dnn.readNetFromTorch(ruta_modelo)
        print("[OK] Modelo cargado correctamente.")
        return red_neural
    except cv2.error as e:
        print(f"[ERROR] No se pudo cargar el modelo FaceNet.")
        exit()

def obtener_embedding(red_neural, imagen_rostro: np.ndarray) -> np.ndarray:
    # El modelo FaceNet que se usa aqui espera una entrada de 96x96
    # La normalizacion (1.0/255) es clave para FaceNet
    blob = cv2.dnn.blobFromImage(
        imagen_rostro, 1.0/255, (96, 96), 
        (0, 0, 0), swapRB=True, crop=False
    )
    
    red_neural.setInput(blob)
    embedding = red_neural.forward()
    
    return embedding.flatten()

def procesar_dataset(red_neural, carpeta_dataset: str, archivo_salida: str):
    datos_embeddings: Dict[str, Any] = {
        "embeddings": [],
        "etiquetas": [], # El indice numerico que representa el nombre
        "nombres": []
    }

    # Buscar subcarpetas (usuarios)
    rutas_usuarios: List[str] = [d for d in os.listdir(carpeta_dataset) 
                                 if os.path.isdir(os.path.join(carpeta_dataset, d))]

    if not rutas_usuarios:
        print("[ERROR] No se encontraron carpetas de usuarios en:", carpeta_dataset)
        return

    print(f"\n[INFO] Usuarios encontrados: {rutas_usuarios}")
    print("[INFO] Generando embeddings...\n")

    imagenes_totales = 0
    imagenes_fallidas = 0

    for nombre_usuario in rutas_usuarios:
        ruta_usuario = os.path.join(carpeta_dataset, nombre_usuario)
        imagenes = [f for f in os.listdir(ruta_usuario) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        
        print(f"Procesando usuario: {nombre_usuario} ({len(imagenes)} imagenes)")
        
        # Asignar una nueva etiqueta numerica si es un nuevo usuario
        if nombre_usuario not in datos_embeddings["nombres"]:
            datos_embeddings["nombres"].append(nombre_usuario)
        etiqueta_numerica = datos_embeddings["nombres"].index(nombre_usuario)

        for nombre_img in imagenes:
            ruta_img = os.path.join(ruta_usuario, nombre_img)
            # Leer imagen
            img = cv2.imread(ruta_img)
            if img is None:
                print(f"Error leyendo: {nombre_img}")
                imagenes_fallidas += 1
                continue
            
            # El modelo FaceNet requiere la imagen pre-redimensionada para que el 
            # blobFromImage pueda trabajar correctamente (aunque 192x192 no es 96x96,
            # el pre-resize ayuda a estandarizar la entrada del dataset).
            # blobFromImage se encarga de cambiarla a 96x96
            img = cv2.resize(img, (192, 192))
            try:
                # Generar embedding
                embedding = obtener_embedding(red_neural, img)
                # Guardar datos
                datos_embeddings["embeddings"].append(embedding)
                datos_embeddings["etiquetas"].append(etiqueta_numerica)
                imagenes_totales += 1
                
            except Exception as e:
                print(f"Error procesando {nombre_img}: {e}")
                imagenes_fallidas += 1
        
        # Mostrar el progreso por usuario
        print(f"{imagenes_totales} embeddings generados hasta ahora (incluye anteriores)")

    # CONVERTIR A NUMPY Y GUARDAR
    datos_embeddings["embeddings"] = np.array(datos_embeddings["embeddings"])
    datos_embeddings["etiquetas"] = np.array(datos_embeddings["etiquetas"])

    print(f"\n{'='*50}")
    print(f"RESUMEN FINAL:")
    print(f"{'='*50}")
    print(f"Total de imagenes procesadas: {imagenes_totales}")
    print(f"Imagenes fallidas: {imagenes_fallidas}")
    print(f"Usuarios registrados: {len(datos_embeddings['nombres'])}")
    print(f"Nombres (Etiquetas): {datos_embeddings['nombres']}")
    print(f"Shape de embeddings: {datos_embeddings['embeddings'].shape}")
    print(f"{'='*50}\n")

    # Guardar archivo
    with open(archivo_salida, 'wb') as f:
        pickle.dump(datos_embeddings, f)
    print(f"[OK] Embeddings guardados en: {archivo_salida}")

if __name__ == "__main__":
    # 1. Cargar el modelo FaceNet
    red_neural_cargada = cargar_modelo(FACENET_MODELO)
    # 2. Procesar el dataset
    procesar_dataset(red_neural_cargada, CARPETA_DATASET, ARCHIVO_SALIDA)
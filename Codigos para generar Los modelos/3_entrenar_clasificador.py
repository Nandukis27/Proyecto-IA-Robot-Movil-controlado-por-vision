import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any

ARCHIVO_EMBEDDINGS = "embeddings_faciales.pkl" # Carga los embeddings 
MODELO_SALIDA = "clasificador_facial.pkl"
TAMANO_PRUEBA = 0.2  # 20% para pruebas, 80% para entrenamiento

def cargar_datos_embeddings(ruta_archivo: str) -> Dict[str, Any]:
    """
    Carga los embeddings, etiquetas y nombres guardados en el archivo pickle.
    """
    print("[INFO] Cargando embeddings...")
    try:
        with open(ruta_archivo, 'rb') as f:
            datos = pickle.load(f)

        embeddings = datos["embeddings"]
        # Usamos 'etiquetas' en lugar de 'labels' si se uso el script anterior
        etiquetas = datos["etiquetas"] 
        nombres = datos["nombres"]
        
        print(f"[OK] Embeddings cargados correctamente")
        print(f"     Total de muestras: {len(embeddings)}")
        print(f"     Usuarios: {nombres}")
        print(f"     Shape: {embeddings.shape}\n")
        
        return embeddings, etiquetas, nombres
        
    except FileNotFoundError:
        print(f"[ERROR] No se encontro el archivo {ruta_archivo}")
        print("        Asegurate de ejecutar el script de generacion de embeddings primero.")
        exit()
    except KeyError:
        print(f"[ERROR] El archivo {ruta_archivo} no contiene las claves esperadas (embeddings, etiquetas, nombres).")
        exit()


def entrenar_y_evaluar_clasificador(embeddings, etiquetas, nombres, tamano_prueba: float):
    """
    Normaliza, divide el dataset, entrena el clasificador SVM y evalua el resultado.
    """
    # 1. NORMALIZAR EMBEDDINGS
    print("[INFO] Normalizando embeddings...")
    normalizador = Normalizer(norm='l2')
    embeddings_normalizados = normalizador.transform(embeddings)

    # 2. CODIFICAR ETIQUETAS
    # Usamos LabelEncoder para convertir las etiquetas numericas originales 
    # (por ejemplo, 0, 1, 2) a un rango consecutivo de indices para el clasificador.
    codificador_etiquetas = LabelEncoder()
    etiquetas_codificadas = codificador_etiquetas.fit_transform(etiquetas)

    print(f"[INFO] Clases detectadas (Indices originales): {codificador_etiquetas.classes_}")
    
    # Mostrar el mapeo (indice_codificado -> nombre)
    # Aqui asumimos que el array 'nombres' tiene los nombres en el orden de las etiquetas originales (0, 1, 2...)
    # Por ejemplo: {0: 'fernando', 1: 'juan', 2: 'maria'}
    mapeo_nombres = {indice: nombres[etiqueta_original] 
                     for indice, etiqueta_original in enumerate(codificador_etiquetas.classes_)}
    print(f"       Mapeo de indices: {mapeo_nombres}\n")
    
    # 3. DIVIDIR DATASET
    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
        embeddings_normalizados, etiquetas_codificadas, 
        test_size=tamano_prueba, 
        random_state=42,
        stratify=etiquetas_codificadas # Asegura que la distribucion de clases sea igual en ambos sets
    )

    print(f"[INFO] Dataset dividido:")
    print(f"       Entrenamiento: {len(X_entrenamiento)} muestras")
    print(f"       Prueba: {len(X_prueba)} muestras\n")

    # 4. ENTRENAR SVM
    print("[INFO] Entrenando clasificador SVM...")
    
    modelo_svm = SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        probability=True,  # Necesario para obtener la confianza de la prediccion
        random_state=42
    )

    modelo_svm.fit(X_entrenamiento, y_entrenamiento)

    print("[OK] Clasificador entrenado correctamente!\n")

    # 5. EVALUAR MODELO
    print("="*60)
    print("EVALUACION DEL MODELO")
    print("="*60)

    y_prediccion = modelo_svm.predict(X_prueba)

    precision = accuracy_score(y_prueba, y_prediccion)
    print(f"\nPrecision general: {precision*100:.2f}%\n")

    # Reporte de clasificacion
    print("Reporte detallado:")
    print("-"*60)
    # Obtenemos los nombres de las clases usando el orden del codificador
    nombres_objetivo = [nombres[i] for i in codificador_etiquetas.classes_]
    print(classification_report(y_prueba, y_prediccion, target_names=nombres_objetivo))

    # 6. MATRIZ DE CONFUSION
    matriz_confusion = confusion_matrix(y_prueba, y_prediccion)

    plt.figure(figsize=(8, 6))
    sns.heatmap(matriz_confusion, annot=True, fmt='d', cmap='Blues', 
                xticklabels=nombres_objetivo, 
                yticklabels=nombres_objetivo)
    plt.title('Matriz de Confusion')
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Prediccion del Modelo')
    plt.tight_layout()
    plt.savefig('matriz_confusion.png', dpi=150)
    print("[OK] Matriz de confusion guardada: matriz_confusion.png\n")
    # 
    return modelo_svm, codificador_etiquetas, normalizador, precision, nombres


def probar_predicciones(modelo, codificador, nombres, X_prueba, y_prueba):
    """
    Realiza y muestra predicciones de muestra con su nivel de confianza.
    """
    print("="*60)
    print("PROBANDO PREDICCIONES DE MUESTRA")
    print("="*60)

    # Seleccionar hasta 5 muestras aleatorias del conjunto de prueba
    num_muestras = min(5, len(X_prueba))
    indices_muestras = np.random.choice(len(X_prueba), num_muestras, replace=False)

    for idx in indices_muestras:
        muestra = X_prueba[idx].reshape(1, -1)
        
        # Obtener el nombre real (decodificando el indice codificado de y_prueba)
        etiqueta_real_indice_original = codificador.inverse_transform([y_prueba[idx]])[0]
        etiqueta_real = nombres[etiqueta_real_indice_original]
        
        # Prediccion del modelo (indice codificado)
        pred_label_idx_codificado = modelo.predict(muestra)[0]
        
        # Decodificar la prediccion a nombre
        pred_label_idx_original = codificador.inverse_transform([pred_label_idx_codificado])[0]
        pred_label = nombres[pred_label_idx_original]
        
        # Probabilidades y confianza
        probabilidades = modelo.predict_proba(muestra)[0]
        confianza = np.max(probabilidades) * 100
        
        estado = "✅" if etiqueta_real == pred_label else "❌"
        
        print(f"\n{estado} Real: {etiqueta_real} | Prediccion: {pred_label} | Confianza: {confianza:.1f}%")


if __name__ == "__main__":
    
    # 1. Cargar datos
    embeddings, etiquetas, nombres = cargar_datos_embeddings(ARCHIVO_EMBEDDINGS)

    # 2. Entrenar, evaluar y obtener resultados
    modelo_final, codificador, normalizador, precision, nombres_entrenados = \
        entrenar_y_evaluar_clasificador(embeddings, etiquetas, nombres, TAMANO_PRUEBA)
        
    # 3. Probar predicciones de muestra
    # Es necesario volver a dividir los datos para tener X_prueba y y_prueba
    _, X_prueba, _, y_prueba = train_test_split(
        normalizador.transform(embeddings), codificador.fit_transform(etiquetas), 
        test_size=TAMANO_PRUEBA, 
        random_state=42,
        stratify=codificador.fit_transform(etiquetas)
    )
    
    probar_predicciones(modelo_final, codificador, nombres_entrenados, X_prueba, y_prueba)

    # 4. GUARDAR MODELO
    print("\n" + "="*60)
    print("GUARDANDO MODELO")
    print("="*60)

    datos_modelo = {
        "modelo": modelo_final,
        "codificador_etiquetas": codificador,
        "normalizador": normalizador,
        "nombres": nombres_entrenados,
        "precision": precision
    }

    with open(MODELO_SALIDA, 'wb') as f:
        pickle.dump(datos_modelo, f)

    print(f"\n[OK] Modelo guardado en: {MODELO_SALIDA}")
    print(f"     Precision: {precision*100:.2f}%")
    print(f"     Usuarios entrenados: {nombres_entrenados}")

    # 5. RESUMEN FINAL
    print("\n" + "="*60)
    print("¡ENTRENAMIENTO COMPLETADO!")
    print("="*60)
    print(f"""
Estadisticas:
   • Total de muestras: {len(embeddings)}
   • Precision alcanzada: {precision*100:.2f}%
   • Usuarios registrados: {len(nombres_entrenados)}
   
Archivos generados:
   • {MODELO_SALIDA} (modelo entrenado)
   • matriz_confusion.png (visualizacion)
""")
    print("="*60)
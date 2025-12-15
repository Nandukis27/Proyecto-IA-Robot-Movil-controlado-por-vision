import numpy as np
import pickle
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict, Any

# ============================
# ⚙️ CONFIGURACION
# ============================
ARCHIVO_DATASET = "dataset_manos/gestos_manos_dataset.npz"
ARCHIVO_INFO = "dataset_manos/dataset_info.json"
MODELO_SALIDA = "clasificador_gestos.pkl"
TAMANO_PRUEBA = 0.2
ESTADO_ALEATORIO = 42

# ============================
# 🛠️ FUNCIONES AUXILIARES
# ============================

def cargar_datos_y_info(ruta_dataset: str, ruta_info: str):
    """Carga los landmarks (X), etiquetas (y) y la información del dataset."""
    print("[INFO] Cargando dataset de gestos...")
    try:
        # Cargar datos
        datos = np.load(ruta_dataset)
        X = datos['landmarks']
        y = datos['labels']
        
        # Cargar información
        with open(ruta_info, 'r') as f:
            info = json.load(f)
        
        print(f"[OK] Dataset cargado correctamente")
        print(f"     Total de muestras: {len(X)}")
        print(f"     Caracteristicas por muestra: {X.shape[1]}")
        print(f"     Gestos: {list(info['gestos'].values())}\n")
        
        return X, y, info
        
    except FileNotFoundError:
        print(f"[ERROR] No se encontro el archivo {ruta_dataset}")
        print("        Asegurate de ejecutar el script de captura de gestos (4_capturar_gestos.py).")
        exit()

def entrenar_y_evaluar_rf(X, y_codificada, nombres_clases):
    """Divide, entrena el modelo Random Forest y realiza la validación cruzada."""
    
    # 1. DIVIDIR DATASET
    print("[INFO] Dividiendo dataset...")
    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
        X, y_codificada,
        test_size=TAMANO_PRUEBA,
        random_state=ESTADO_ALEATORIO,
        stratify=y_codificada
    )

    print(f"[OK] Dataset dividido:")
    print(f"     Entrenamiento: {len(X_entrenamiento)} muestras ({(1-TAMANO_PRUEBA)*100:.0f}%)")
    print(f"     Prueba: {len(X_prueba)} muestras ({TAMANO_PRUEBA*100:.0f}%)\n")

    # 2. ENTRENAR RANDOM FOREST
    print("[INFO] Entrenando clasificador Random Forest...")

    modelo = RandomForestClassifier(
        n_estimators=200,        
        max_depth=20,            
        min_samples_split=5,     
        min_samples_leaf=2,      
        random_state=ESTADO_ALEATORIO,
        n_jobs=-1                
    )

    modelo.fit(X_entrenamiento, y_entrenamiento)
    print("[OK] Clasificador entrenado correctamente!\n")

    # 3. VALIDACION CRUZADA
    print("[INFO] Realizando validacion cruzada (5-fold)...")
    puntuaciones_cv = cross_val_score(modelo, X_entrenamiento, y_entrenamiento, cv=5, scoring='accuracy')

    print(f"[OK] Resultados de validacion cruzada:")
    print(f"     Precision promedio: {puntuaciones_cv.mean()*100:.2f}% (±{puntuaciones_cv.std()*100:.2f}%)")
    print(f"     Scores individuales: {[f'{s*100:.2f}%' for s in puntuaciones_cv]}\n")
    
    return modelo, X_prueba, y_prueba, puntuaciones_cv

def evaluar_modelo(modelo, X_prueba, y_prueba, nombres_clases):
    """Genera métricas de evaluación, el reporte detallado y la matriz de confusión."""
    
    print("="*60)
    print("EVALUACION EN CONJUNTO DE PRUEBA")
    print("="*60)

    y_prediccion = modelo.predict(X_prueba)
    precision = accuracy_score(y_prueba, y_prediccion)

    print(f"\n🎯 Precision en test: {precision*100:.2f}%\n")

    # Reporte detallado
    print("📋 Reporte de clasificacion:")
    print("-"*60)
    print(classification_report(y_prueba, y_prediccion, target_names=nombres_clases))

    # MATRIZ DE CONFUSION
    matriz_confusion = confusion_matrix(y_prueba, y_prediccion)

    plt.figure(figsize=(10, 8))
    sns.heatmap(matriz_confusion, annot=True, fmt='d', cmap='Blues',
                xticklabels=nombres_clases,
                yticklabels=nombres_clases,
                cbar_kws={'label': 'Numero de muestras'})
    plt.title('Matriz de Confusion - Reconocimiento de Gestos', fontsize=14, pad=20)
    plt.ylabel('Real', fontsize=12)
    plt.xlabel('Prediccion', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('matriz_confusion_gestos.png', dpi=150, bbox_inches='tight')
    print("[OK] Matriz de confusion guardada: matriz_confusion_gestos.png\n")
    
    return precision

def analizar_importancia(modelo):
    """Calcula y grafica la importancia de las características (landmarks)."""
    print("="*60)
    print("📈 ANALISIS DE IMPORTANCIA DE CARACTERISTICAS")
    print("="*60)

    importancia_caracteristicas = modelo.feature_importances_
    top_n = 10
    top_indices = np.argsort(importancia_caracteristicas)[-top_n:][::-1]

    print(f"\nTop {top_n} caracteristicas mas importantes:")
    for i, idx in enumerate(top_indices, 1):
        # El indice de la caracteristica (0-62) se mapea a punto (0-20) y coordenada (x, y, z)
        punto_idx = idx // 3
        coord = ['x', 'y', 'z'][idx % 3]
        print(f"  {i}. Punto {punto_idx} ({coord}): {importancia_caracteristicas[idx]:.4f}")

    # Visualizar importancia
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(importancia_caracteristicas)), importancia_caracteristicas, alpha=0.7)
    plt.xlabel('Indice de caracteristica')
    plt.ylabel('Importancia')
    plt.title('Importancia de Caracteristicas (landmarks)')
    plt.tight_layout()
    plt.savefig('importancia_caracteristicas.png', dpi=150)
    print("\n[OK] Grafico de importancia guardado: importancia_caracteristicas.png\n")

def probar_muestras_aleatorias(modelo, X_prueba, y_prueba, codificador):
    """Realiza predicciones de muestra para verificar confianza."""
    print("="*60)
    print("🧪 PROBANDO PREDICCIONES ALEATORIAS")
    print("="*60)

    n_muestras = min(10, len(X_prueba))
    sample_indices = np.random.choice(len(X_prueba), n_muestras, replace=False)

    correctas = 0
    for idx in sample_indices:
        muestra = X_prueba[idx].reshape(1, -1)
        etiqueta_real = codificador.inverse_transform([y_prueba[idx]])[0]
        
        etiqueta_predicha_idx = modelo.predict(muestra)[0]
        etiqueta_predicha = codificador.inverse_transform([etiqueta_predicha_idx])[0]
        
        proba = modelo.predict_proba(muestra)[0]
        confianza = max(proba) * 100
        
        status = "✅" if etiqueta_real == etiqueta_predicha else "❌"
        if etiqueta_real == etiqueta_predicha:
            correctas += 1
        
        print(f"{status} Real: {etiqueta_real:15s} | Prediccion: {etiqueta_predicha:15s} | Confianza: {confianza:5.1f}%")

    print(f"\n📊 Precision en muestras: {correctas}/{n_muestras} ({correctas/n_muestras*100:.1f}%)")

def guardar_resultado_final(modelo, codificador, info_gestos, precision, puntuaciones_cv, num_caracteristicas):
    """Guarda el modelo entrenado y sus estadísticas."""
    
    datos_modelo: Dict[str, Any] = {
        "modelo": modelo,
        "codificador_etiquetas": codificador,
        "gestos": info_gestos,
        "precision_test": precision,
        "cv_mean": puntuaciones_cv.mean(),
        "cv_std": puntuaciones_cv.std(),
        "num_caracteristicas": num_caracteristicas
    }

    with open(MODELO_SALIDA, 'wb') as f:
        pickle.dump(datos_modelo, f)

    print("\n" + "="*60)
    print("💾 GUARDANDO MODELO")
    print("="*60)

    print(f"\n[OK] Modelo guardado en: {MODELO_SALIDA}")
    print(f"     Precision en test: {precision*100:.2f}%")
    print(f"     Precision en CV: {puntuaciones_cv.mean()*100:.2f}% (±{puntuaciones_cv.std()*100:.2f}%)")
    print(f"     Gestos reconocidos: {list(codificador.classes_)}")

# ============================
# 🎬 FUNCION PRINCIPAL
# ============================

def iniciar_entrenamiento():
    
    print("="*60)
    print("🤖 ENTRENAMIENTO DE CLASIFICADOR DE GESTOS")
    print("="*60)
    
    # 1. CARGAR Y CODIFICAR DATOS
    X, y, info = cargar_datos_y_info(ARCHIVO_DATASET, ARCHIVO_INFO)
    
    print("[INFO] Codificando etiquetas...")
    codificador = LabelEncoder()
    y_codificada = codificador.fit_transform(y)
    nombres_clases = codificador.classes_
    info_gestos = info['gestos']
    
    print(f"[OK] Clases detectadas: {nombres_clases}")
    
    # 2. ENTRENAR Y VALIDAR
    modelo_rf, X_prueba, y_prueba, puntuaciones_cv = entrenar_y_evaluar_rf(
        X, y_codificada, nombres_clases
    )

    # 3. EVALUAR Y GRAFICAR MATRIZ DE CONFUSION
    precision_test = evaluar_modelo(modelo_rf, X_prueba, y_prueba, nombres_clases)

    # 4. ANALIZAR E IMPORTANCIA DE CARACTERISTICAS
    analizar_importancia(modelo_rf)

    # 5. PROBAR MUESTRAS
    probar_muestras_aleatorias(modelo_rf, X_prueba, y_prueba, codificador)

    # 6. GUARDAR RESULTADO FINAL
    guardar_resultado_final(
        modelo_rf, codificador, info_gestos, precision_test, puntuaciones_cv, X.shape[1]
    )

    # 7. RESUMEN FINAL
    print("\n" + "="*60)
    print("🎉 ¡ENTRENAMIENTO COMPLETADO!")
    print("="*60)
    print(f"""
Estadisticas finales:
   • Total de muestras: {len(X)}
   • Precision en test: {precision_test*100:.2f}%
   • Precision en CV: {puntuaciones_cv.mean()*100:.2f}%
   • Gestos entrenados: {len(nombres_clases)}
   
Archivos generados:
   • {MODELO_SALIDA} (modelo entrenado)
   • matriz_confusion_gestos.png
   • importancia_caracteristicas.png

""")
    print("="*60)


# Ejecutar la funcion principal
if __name__ == "__main__":
    iniciar_entrenamiento()
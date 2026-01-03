# 🧬 ProjectEvo: Optimización Neuro-Evolutiva para Seguridad Ciudadana

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![PyGAD](https://img.shields.io/badge/AI-Evolutionary-green)
![Status](https://img.shields.io/badge/Status-Finalizado-success)

> **Tesis de Maestría:** Implementación de un sistema híbrido (Deep Learning + Algoritmos Genéticos) para la clasificación de incidentes críticos en escenarios de alto desbalance de datos (Callao, Perú).

---

## 📋 Descripción del Proyecto

Este repositorio contiene el código fuente y los experimentos desarrollados para el curso **Computación Evolutiva (MAE-IA 2025-II)** de la **Universidad Nacional de Ingeniería (UNI)**.

El objetivo principal es resolver el problema de clasificación de textos cortos (reportes de seguridad) donde existe un desbalance extremo: miles de reportes rutinarios ("Ambientales") frente a escasos eventos críticos ("Protección Familiar", "Física").

**Solución Propuesta:** Un pipeline que integra **BERT** para la extracción de características semánticas y **Algoritmos Genéticos (GA)** para optimizar tanto la selección de características como los hiperparámetros de entrenamiento (especialmente los pesos de la función de pérdida), culminando en un **Ensemble Evolutivo**.

---

## 🚀 Pipeline de Ingeniería (Metodología)

El proyecto se divide en 5 notebooks secuenciales que representan las fases de investigación:

### 🔹 Fase 1: Línea Base y Representación (`01_Baseline` & `02_BERT`)
* Transformación de textos crudos a vectores densos usando **BERT Multilingual**.
* Generación de embeddings de 768 dimensiones que capturan el contexto semántico.
* *Resultado:* Dataset procesado y fragmentado para manejo eficiente de memoria.

### 🔹 Fase 2: Selección Evolutiva de Características (`03_GA_FeatureSelection`)
* Uso de un Algoritmo Genético para reducir la dimensionalidad.
* **Logro:** Se redujo el espacio de entrada de 768 a **92 características clave** (reducción del 88%), eliminando ruido y mejorando la velocidad de entrenamiento sin sacrificar precisión.

### 🔹 Fase 3: Validación de Arquitectura (`04_NAS_NeuroEvolution`)
* Comparación empírica ("Architecture Showdown") de topologías neuronales.
* **Decisión:** Selección de una arquitectura **3 capas ocultas x 512 neuronas** basada en el Principio de Parsimonia, superando a modelos más superficiales y masivos.

### 🔹 Fase 4: Optimización de Hiperparámetros (`05_Evolutionary_Ensemble`)
* Ajuste fino evolutivo de la "química" del entrenamiento: *Learning Rate*, *Dropout* y *Alpha* (Weighted Loss).
* **Innovación:** El GA descubrió que penalizar **5 veces más (Alpha=5.0)** los errores en clases críticas era la clave para romper la barrera del desbalance.

### 🔹 Fase 5: Producción (`05_Evolutionary_Ensemble`)
* Despliegue de un **Comité de Expertos (Ensemble)** de 5 modelos entrenados con semillas distintas (Bagging).
* Inferencia final mediante votación ponderada.

---

## 📂 Estructura del Repositorio

```text
ProjectEvo-Callao-Safety/
│
├── data/                   # Datos crudos
├── data_processed/         # Embeddings de BERT fragmentados (.pt)
├── models/                 # Artefactos entrenados
│   ├── ga_feature_mask.pt  # Máscara binaria (92 features)
│   └── production_final.pt # Pesos del Ensemble final
│
├── notebooks/
│   ├── 01_Baseline_Clasico.ipynb       # Modelos tradicionales (SVM, RF)
│   ├── 02_BERT_Embeddings.ipynb        # Extracción de features con Transformers
│   ├── 03_GA_FeatureSelection.ipynb    # Reducción de dimensionalidad con PyGAD
│   ├── 04_NAS_NeuroEvolution.ipynb     # Búsqueda de arquitectura y validación
│   └── 05_Evolutionary_Ensemble.ipynb  # Tuning, Ensemble y Evaluación Final
│
├── requirements.txt        # Dependencias del proyecto
└── README.md               # Este archivo

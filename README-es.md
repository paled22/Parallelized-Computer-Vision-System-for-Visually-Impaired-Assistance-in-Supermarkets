# Sistema de Visión Computacional Optimizado para Asistencia en Supermercados
🌍 *Read this in [English](README.md)*

## Resumen
Este repositorio contiene la prueba de concepto de un pipeline de visión computacional diseñado para asistir a personas con discapacidad visual a localizar productos en estanterías de supermercado. El proyecto explora el equilibrio (trade-off) entre precisión, velocidad de inferencia y tamaño del modelo al adaptar una arquitectura YOLOv8n estándar para su potencial despliegue en Edge Computing.

## Arquitectura del Sistema
El sistema utiliza un pipeline en cascada:
* **Detección (Paralelismo de Arquitectura):** Un modelo YOLOv8 modificado que reemplaza las convoluciones estándar por convoluciones separables por profundidad (YOLO-DWConv). Este paralelismo matemático procesa los canales de entrada de forma independiente, optimizando los recursos de hardware.
* **Clasificación:** Un modelo YOLOv8-cls que clasifica los recortes generados (regiones de interés) en 25 categorías de productos.
* **Módulo de Accesibilidad:** Integración con la librería `gTTS` para transformar las coordenadas espaciales de las cajas delimitadoras en instrucciones de audio direccionales (ej. "izquierda", "centro", "derecha").

## Resultados Principales
* **Eficiencia y Tamaño:** Las modificaciones arquitectónicas con DWConv lograron reducir los parámetros totales en un 18.8% (de 3,005,843 a 2,440,211) y la carga computacional de 8.1 a 7.2 GFLOPs.
* **Trade-off de Concurrencia y Precisión:** Se implementó concurrencia a nivel de software utilizando `ThreadPoolExecutor` de Python para el procesamiento de lotes de imágenes. Se logró una mejora del 45.3% en la velocidad de inferencia, documentando la sobrecarga esperada por el manejo de hilos (GIL) y una caída en la precisión mAP50 de 0.8173 a 0.6840.
* **Preparación para Móviles (Edge):** Se validó la viabilidad de despliegue en dispositivos móviles mediante una cuantización preliminar a INT8 usando LiteRT/TFLite, logrando una compresión del modelo de ~3.6x.

## Dataset
Este proyecto fue entrenado y evaluado utilizando el **Freiburg Groceries Dataset**.
*(Nota: Por buenas prácticas, los datasets no están incluidos en este repositorio. Por favor, descárgalo de manera independiente y colócalo en el directorio `/data` antes de ejecutar el código).*

## Tecnologías
Python, PyTorch, Ultralytics YOLO, OpenCV, `concurrent.futures`, Google Colab, TFLite.

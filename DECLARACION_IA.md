# Declaración de uso de herramientas generativas (BORRADOR)

Herramienta: Claude Code (Anthropic), modelo Claude Opus 5.5.

> Borrador para que el grupo lo revise y complete. Debe reflejar lo que realmente ocurrió
> al momento de entregar: actualizar esta lista si cambia lo que hizo la herramienta o lo
> que modificó el grupo.

## Para qué se usó

- Auditoría técnica del dataset: conteos, emparejamiento de imágenes y labels, y búsqueda de imágenes casi duplicadas.
- Programación del código en `src/`: preparación de particiones, carga de datos, ciclos de entrenamiento, evaluador común, medición de latencia y generación de tablas y gráficos sin interpretación.
- Implementación en código del detector FCOS según el diseño definido por el grupo (`src/custom_cnn.py`).
- Medición de memoria y velocidad de la GPU para las combinaciones consideradas (`artifacts/gpu_probe*.csv`).
- Depuración del código y pruebas del evaluador y de FCOS (`tests/`).
- Ejecución de los entrenamientos con las configuraciones definidas por el grupo.
- Al pedir los valores de diseño, la herramienta presentó opciones concretas (dentro de los rangos que dio el grupo y con los datos de memoria y tiempo medidos) y el grupo eligió entre ellas. [REVISAR: indicar qué valores se eligieron así.]
- Preparación del repositorio git.

## Para qué no se usó

Conforme al enunciado, las siguientes tareas fueron realizadas por el grupo:

- Diseño de la solución: arquitectura de la CNN propia, elección de modelos, hiperparámetros, número de épocas, regularización, aumento de datos, congelamiento y fine-tuning, partición del dataset y tratamiento de las clases.
- Justificación de esas decisiones.
- Análisis, interpretación y comparación de los resultados.

## Modificaciones del grupo a la implementación

[COMPLETAR: qué archivos y qué partes modificó el grupo.]

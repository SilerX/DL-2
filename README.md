# Tarea 2 - Detección de objetos en FjordFish

Todos los modelos comparten las mismas particiones y el mismo evaluador (`src/detection_eval.py`, basado en pycocotools).

## Dataset

FjordFish ([Zenodo, registro 17496689](https://zenodo.org/records/17496689)); si se usa, debe citarse según esa página. Las imágenes y los pesos no están en el repositorio: `data/` se regenera con los comandos de la sección Particiones (`<FjordFish>` es la carpeta del dataset descargado) y los pesos COCO se descargan solos al entrenar. Sí se incluyen las listas de ids de cada partición (`data/fjordfish/splits/`) y `data/fjordfish/manifest.json`.

## Entorno

```bash
python -m venv .venv --system-site-packages
.venv/Scripts/python -m pip install -r requirements.txt
```

Requiere PyTorch con CUDA (usado: torch 2.6.0+cu124, torchvision 0.21.0). La GTX 1650 no usa precisión mixta (`amp: false` en YOLO): la serie GTX 16xx tiene problemas conocidos con AMP.

## Flujo

1. **Revisar duplicados**: `python src/check_duplicates.py --root <FjordFish> --out artifacts/audit --threshold <n>`
2. **Crear particiones**: `python src/prepare_dataset.py --root <FjordFish> --out data/<nombre> --cache data/cache_<lado> --strategy <keep_val|regroup> --test-frac <f> [--train-frac <f> --val-frac <f>] [--duplicates artifacts/audit/duplicate_groups.json] [--class-map <yaml>] --max-side <n> --seed <n>`
3. **Completar configuraciones**: `configs/*.yaml`. Cada `null` es una decisión pendiente y el entrenamiento se niega a partir hasta completarla.
4. **Entrenar**:
   - `python src/train_torch.py --config configs/cnn_scratch.yaml`
   - `python src/train_yolo.py --config configs/yolo_scratch.yaml`
   - `python src/train_yolo.py --config configs/yolo_coco.yaml`
   - `python src/train_torch.py --config configs/frcnn_coco.yaml`
5. **Evaluar**: `python src/evaluate_run.py --run runs/<nombre> --data data/<nombre>`; al final, `python src/remeasure_latency.py --run runs/<nombre> --data data/<nombre>`
6. **Comparar**: `python src/make_report.py --runs runs/a runs/b --labels "A" "B" --out results`

## Medición

- **Umbral de confianza**: se elige en val (mayor F1 micro, IoU 0.5) y se aplica sin cambios en test.
- **mAP**: COCO, promedio solo sobre las clases con cajas reales en la partición evaluada.
- **Clases por caja**: todos los modelos pueden devolver una misma caja con más de una clase, como hacen `model.val()` de ultralytics y Faster R-CNN. `predict()` de ultralytics deja solo una, así que `src/predictors.py` lo ajusta.
- **Máximo de detecciones**: 100 por imagen en todos los modelos.
- **Tamaños pequeño/mediano/grande**: áreas COCO en píxeles, medidas sobre las imágenes ya reducidas.
- **Latencia**: una imagen a la vez, incluye lectura, preproceso, red y postproceso; descarta 10 imágenes de calentamiento y mide 100. La tabla final usa `src/remeasure_latency.py`: con la GPU sin otros procesos, repite esa medición 3 veces y guarda la mediana (`latency_ms`), las repeticiones (`latency_ms_runs`) y la medición original de `evaluate_run.py` (`latency_ms_previous`) en `evaluation.json`.
- **Tiempo de entrenamiento**: reloj total, incluida la validación por época.
- **Selección del mejor modelo**: los modelos PyTorch usan mAP@0.5:0.95 en val; ultralytics usa su métrica `fitness`. La comparación final siempre usa el evaluador común.

## Congelamiento

`runs/<nombre>/freeze_report.json` registra los parámetros congelados y entrenables por capa o grupo de cada ejecución.

## Particiones

`data/<nombre>/splits/*.txt` contiene los ids de cada partición y `data/<nombre>/manifest.json`, los argumentos usados y los conteos por clase.

El dataset final se generó con:

```bash
python src/check_duplicates.py --root <FjordFish> --out artifacts/audit/final --threshold 14 --consecutive-limit 20
python src/prepare_dataset.py --root <FjordFish> --out data/fjordfish --cache data/cache_1333 --strategy regroup --train-frac 0.8 --val-frac 0.1 --test-frac 0.1 --duplicates artifacts/audit/final/duplicate_groups.json --class-map configs/classes.yaml --max-side 1333 --seed 0
python src/verify_split.py --data data/fjordfish --root <FjordFish> --out artifacts/split_check
```

- **Partición original descartada:** `train` y `val` originales compartían frames de las mismas escenas.
- **Umbral 14:** con 6, 8, 10 y 12 aún aparecían frames de la misma escena entre particiones (revisión visual); desde 16 son mayormente escenas distintas.
- **Límite de ids consecutivos 20:** une frames seguidos de un mismo clip que el umbral no alcanza. La distancia entre ids consecutivos tiene dos poblaciones, con un valle cerca de 20; los pares al azar se concentran en 30-32.
- **Limitación:** el dataset no trae un identificador de video, así que no se puede garantizar que no quede ninguna filmación repartida. Frames de un mismo montaje con otro encuadre pueden quedar en particiones distintas.
- **Lado máximo 1333:** las imágenes se guardan reducidas a ese tamaño, el máximo que usa Faster R-CNN, para acelerar la carga de datos.

## Pruebas

```bash
python tests/test_eval.py
python tests/test_fcos.py
```

`configs/smoke_*.yaml` solo sirve para comprobar que el código corre: no son decisiones de diseño.

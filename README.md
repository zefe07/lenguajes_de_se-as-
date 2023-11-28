# Detector de lenguaje de señas

 Detector de lenguaje de señas con Python, OpenCV y Mediapipe !

## Creacion de enviroment de desarrollo
venv (entorno virtual) es una herramienta que se utiliza en Python para crear entornos virtuales, que son aislamientos de entornos de desarrollo para proyectos específicos. Estos entornos virtuales permiten gestionar las dependencias de un proyecto de manera independiente, evitando conflictos entre las versiones de las bibliotecas utilizadas en diferentes proyectos.

```bash
python3 -m venv venv
source venv/bin/activate
```

## Instalar librerias de python

```bash
pip install -r requirements.txt
```

## Colección de datos

```bash
python collect_imgs.py
```

## Creación de Conjuto de datos

```bash
python create_dataset.py
```

## Entrenamiento del clasifacador con Random Forest

```bash
python train_classifier.py
```

## Inferencia o predición del modelo

```bash
python inference_classifier.py
```




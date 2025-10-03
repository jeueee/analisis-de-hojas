# LeafSegmenter: Analizador de Área Foliar con Detectron2

Una aplicación de escritorio para cargar imágenes de hojas, segmentarlas automáticamente usando el modelo Detectron2, y calcular el área foliar total e individual tras una calibración de escala. Los resultados pueden ser exportados a un archivo Excel junto con las imágenes segmentadas.

## ✨ Características Principales

- **Carga de Imágenes:** Soporta formatos `.jpg`, `.jpeg`, y `.png`.
- **Calibración de Escala:** Calibra la medida en cm usando dos clics sobre una distancia de referencia.
- **Segmentación Automática:** Utiliza el modelo pre-entrenado `mask_rcnn_R_50_FPN_3x` de Detectron2.
- **Cálculo de Área:** Mide el área en cm² de cada segmento y el área total.
- **Exportación de Resultados:** Guarda un reporte en Excel y las imágenes procesadas.

## 🚀 Instalación

1. **Clona este repositorio.**
2. **Crea un entorno virtual** (`python -m venv env`).
3. **Instala Detectron2** siguiendo su [guía de instalación oficial](https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md).
4. **Instala el resto de las dependencias:** `pip install -r requirements.txt`

## ▶️ Cómo Usar

Ejecuta la aplicación con: `python main.py`

## 📄 Cómo Citar
Si utilizas este software en tu investigación, por favor cítalo usando la información del archivo `CITATION.cff`.

## 📜 Licencia
Este proyecto está bajo la Licencia MIT.
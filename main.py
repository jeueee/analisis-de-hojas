import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk
import cv2
import math
import os
from datetime import datetime

# --- Importaciones Adicionales ---
try:
    import pandas as pd
except ImportError:
    messagebox.showerror("Error de Importación", "La biblioteca 'pandas' no está instalada.\nPor favor, instálala con: pip install pandas openpyxl")
    exit()

# Importaciones de Detectron2
try:
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    from detectron2 import model_zoo
    from detectron2.utils.visualizer import Visualizer
    from detectron2.utils.visualizer import ColorMode
    from detectron2.data import MetadataCatalog
except ImportError:
    messagebox.showerror("Error de Importación", "Detectron2 no está instalado. Por favor, instálalo para ejecutar esta aplicación.")
    exit()

# --- Variables Globales ---
predictor = None
metadata = None
original_image = None
pixel_per_cm = None
is_calibrating = False
calibration_points = []

# --- NUEVAS Variables Globales para Exportación ---
export_list = []
processed_image_np = None # Guarda la última imagen procesada (con máscaras)
last_total_area_str = ""
last_individual_areas_str = ""


# --- Widgets Globales ---
root = None
original_panel = None
processed_panel = None
area_label = None
info_label = None
results_text = None
calibration_var = None
add_to_list_button = None # NUEVO
export_button = None      # NUEVO


# --- Configuración del Modelo Detectron2 ---
try:
    print("Cargando modelo Detectron2...")
    cfg = get_cfg()
    cfg.MODEL.DEVICE = "cpu" # Usar CPU para mayor compatibilidad
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    predictor = DefaultPredictor(cfg)
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    print("Modelo cargado exitosamente.")
except Exception as e:
    messagebox.showerror("Error de Carga de Modelo", f"No se pudo cargar el modelo Detectron2: {e}")
    predictor = None

# --- Funciones de la Interfaz ---
def open_image():
    """Abre un cuadro de diálogo para seleccionar una imagen y la muestra."""
    global original_image, pixel_per_cm, calibration_points, processed_image_np
    
    image_path = filedialog.askopenfilename(title="Selecciona una imagen", filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if not image_path:
        return

    original_image = cv2.imread(image_path)
    if original_image is None:
        messagebox.showerror("Error", "No se pudo cargar la imagen.")
        return
        
    display_image(original_image, original_panel)
    # Limpiar estado anterior
    processed_panel.config(image=None)
    processed_panel.image = None
    pixel_per_cm = None
    calibration_points = []
    processed_image_np = None
    area_label.config(text="Área Total Estimada: N/A")
    update_results_text("")
    add_to_list_button.config(state='disabled') # Desactivar botón al cargar nueva imagen

def process_image():
    """Procesa la imagen con Detectron2 y muestra el resultado y las áreas."""
    global processed_image_np
    if original_image is None:
        messagebox.showwarning("Advertencia", "Por favor, carga una imagen primero.")
        return
    
    if predictor is None:
        messagebox.showerror("Error", "El modelo no se cargó correctamente.")
        return

    print("Procesando imagen...")
    outputs = predictor(original_image)
    instances = outputs["instances"].to("cpu")

    v = Visualizer(original_image[:, :, ::-1], metadata=metadata, scale=0.8, instance_mode=ColorMode.SEGMENTATION)
    processed_image_rgb = v.draw_instance_predictions(instances).get_image()
    
    # Guardar la imagen procesada en formato BGR (OpenCV) para exportarla después
    processed_image_np = cv2.cvtColor(processed_image_rgb, cv2.COLOR_RGB2BGR)

    display_image(processed_image_np, processed_panel)

    if pixel_per_cm:
        calculate_and_display_areas(instances)
        add_to_list_button.config(state='normal') # Activar botón tras procesar
    else:
        area_label.config(text="Área Total Estimada: Requiere calibración")
        update_results_text("Calibración necesaria para calcular el área de cada segmento.")
        add_to_list_button.config(state='disabled')
    
    print("Procesamiento completado.")

def calculate_and_display_areas(instances):
    """Calcula el área de cada segmento individualmente y el total."""
    global last_total_area_str, last_individual_areas_str
    
    if not instances.has("pred_masks") or len(instances) == 0:
        area_label.config(text="Área Total Estimada: 0.00 cm²")
        update_results_text("No se encontraron objetos para medir.")
        last_total_area_str = "0.00"
        last_individual_areas_str = "No se encontraron objetos para medir."
        return

    masks = instances.pred_masks
    scores = instances.scores
    pred_classes = instances.pred_classes
    
    total_area_cm2 = 0
    results_content = ""

    for i in range(len(masks)):
        mask = masks[i]
        pixel_area = mask.sum().item()
        area_cm2 = pixel_area / (pixel_per_cm ** 2)
        total_area_cm2 += area_cm2
        
        class_name = metadata.thing_classes[pred_classes[i]]
        score = scores[i]
        
        results_content += f"{i+1}. {class_name}: {area_cm2:.2f} cm² (Conf: {score:.2f})\n"
    
    total_area_str = f"{total_area_cm2:.2f} cm²"
    area_label.config(text=f"Área Total Estimada: {total_area_str}")
    update_results_text(results_content)

    # Guardar resultados para la exportación
    last_total_area_str = f"{total_area_cm2:.2f}"
    last_individual_areas_str = results_content.strip()

def start_calibration():
    """Inicia el proceso de calibración de 2 clics."""
    global is_calibrating, calibration_points
    if original_image is None:
        messagebox.showwarning("Advertencia", "Por favor, carga una imagen primero.")
        return
    
    is_calibrating = True
    calibration_points = []
    info_label.config(text="Modo Calibración: Haz clic en 2 puntos en la imagen para definir la distancia.")

def on_image_click(event):
    """Maneja los clics para la calibración, corrigiendo la posición del punto."""
    global is_calibrating, calibration_points, pixel_per_cm

    if not is_calibrating or original_panel.image is None:
        return

    panel_w, panel_h = original_panel.winfo_width(), original_panel.winfo_height()
    displayed_w, displayed_h = original_panel.image.width(), original_panel.image.height()
    
    offset_x = (panel_w - displayed_w) / 2
    offset_y = (panel_h - displayed_h) / 2

    img_event_x = event.x - offset_x
    img_event_y = event.y - offset_y

    if not (0 <= img_event_x <= displayed_w and 0 <= img_event_y <= displayed_h):
        return

    orig_h, orig_w, _ = original_image.shape
    scale_x, scale_y = orig_w / displayed_w, orig_h / displayed_h
    x_orig, y_orig = int(img_event_x * scale_x), int(img_event_y * scale_y)
    
    calibration_points.append((x_orig, y_orig))

    temp_display_image = original_image.copy()
    for pt in calibration_points:
        cv2.circle(temp_display_image, pt, int(orig_w * 0.01), (0, 0, 255), -1) # Círculo rojo
    display_image(temp_display_image, original_panel)

    if len(calibration_points) == 2:
        is_calibrating = False
        info_label.config(text="")
        
        try:
            known_distance = float(calibration_var.get())
        except (ValueError, TypeError):
            messagebox.showerror("Error", "Introduce un número válido en la distancia de referencia.")
            calibration_points = []
            display_image(original_image, original_panel)
            return
        
        if known_distance > 0:
            p1, p2 = calibration_points
            pixel_distance = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            if pixel_distance > 0:
                pixel_per_cm = pixel_distance / known_distance
                messagebox.showinfo("Calibración Exitosa", f"Escala establecida: {pixel_per_cm:.2f} píxeles por cm.")
            else:
                messagebox.showwarning("Advertencia", "Los puntos están demasiado cerca.")
        else:
            messagebox.showwarning("Advertencia", "La distancia debe ser mayor que cero.")
        
        calibration_points = []
        display_image(original_image, original_panel)

def display_image(image_np, panel):
    """Muestra una imagen en un panel, redimensionándola para que quepa."""
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    
    max_size = (panel.winfo_width() or 500, panel.winfo_height() or 500)
    pil_image.thumbnail(max_size, Image.Resampling.LANCZOS)
    
    tk_image = ImageTk.PhotoImage(pil_image)
    panel.config(image=tk_image)
    panel.image = tk_image

def update_results_text(content):
    """Actualiza el contenido de la casilla de resultados."""
    results_text.config(state='normal')
    results_text.delete('1.0', tk.END)
    results_text.insert(tk.END, content)
    results_text.config(state='disabled')

# --- NUEVAS Funciones de Exportación ---
def add_to_export_list():
    """Añade el resultado actual a la lista de exportación."""
    if processed_image_np is None or not last_total_area_str:
        messagebox.showwarning("Advertencia", "No hay resultados procesados para añadir.")
        return
    
    result_data = {
        "image": processed_image_np.copy(),
        "total_area": last_total_area_str,
        "details": last_individual_areas_str
    }
    export_list.append(result_data)
    
    info_label.config(text=f"{len(export_list)} resultado(s) en la lista de exportación.")
    export_button.config(state='normal')
    add_to_list_button.config(state='disabled') # Desactivar para evitar duplicados

def export_results():
    """Exporta los resultados de la lista a una carpeta con imágenes y un Excel."""
    if not export_list:
        messagebox.showwarning("Advertencia", "La lista de exportación está vacía.")
        return

    # 1. Pedir al usuario que elija una ubicación
    base_path = filedialog.askdirectory(title="Selecciona la carpeta para guardar los resultados")
    if not base_path:
        return

    try:
        # 2. Crear la estructura de carpetas
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        main_folder = os.path.join(base_path, f"Resultados_{timestamp}")
        images_folder = os.path.join(main_folder, "imagenes_segmentadas")
        excel_folder = os.path.join(main_folder, "reporte_excel")

        os.makedirs(images_folder, exist_ok=True)
        os.makedirs(excel_folder, exist_ok=True)

        # 3. Preparar datos para Excel y guardar imágenes
        excel_data = []
        for i, item in enumerate(export_list):
            photo_id = f"foto_{i+1:03d}"
            image_filename = f"{photo_id}.png"
            image_path = os.path.join(images_folder, image_filename)

            # Guardar la imagen segmentada
            cv2.imwrite(image_path, item["image"])

            # Añadir datos a la lista para el DataFrame
            excel_data.append({
                "ID Foto": photo_id,
                "Área Total (cm²)": item["total_area"],
                "Detalles": item["details"]
            })
        
        # 4. Crear y guardar el archivo Excel
        df = pd.DataFrame(excel_data)
        excel_path = os.path.join(excel_folder, "reporte_areas.xlsx")
        df.to_excel(excel_path, index=False, engine='openpyxl')

        messagebox.showinfo("Exportación Exitosa", f"Resultados exportados correctamente en:\n{main_folder}")

        # 5. Limpiar la lista después de exportar
        export_list.clear()
        info_label.config(text="Exportación completa. La lista está vacía.")
        export_button.config(state='disabled')

    except Exception as e:
        messagebox.showerror("Error de Exportación", f"Ocurrió un error al exportar: {e}")


def main():
    """Función principal que crea la GUI y ejecuta la aplicación."""
    global root, original_panel, processed_panel, area_label, info_label, results_text
    global calibration_var, add_to_list_button, export_button
    
    root = tk.Tk()
    root.title("Análisis de Imágenes con Segmentación y Exportación")
    root.geometry("1200x850") # Aumentado un poco la altura para los nuevos botones

    # --- Frame principal ---
    main_frame = tk.Frame(root)
    main_frame.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

    # --- Columna Izquierda (Imagen Original) ---
    left_frame = tk.Frame(main_frame)
    left_frame.pack(side=tk.LEFT, padx=10, fill=tk.BOTH, expand=True)
    tk.Label(left_frame, text="Imagen Original", font=("Helvetica", 12)).pack(pady=(0, 5))
    original_panel = tk.Label(left_frame, relief="sunken", borderwidth=2, bg="gray90")
    original_panel.pack(fill=tk.BOTH, expand=True)
    original_panel.bind("<Button-1>", on_image_click)

    # --- Columna Derecha (Resultados) ---
    right_frame = tk.Frame(main_frame)
    right_frame.pack(side=tk.LEFT, padx=10, fill=tk.BOTH, expand=True)
    tk.Label(right_frame, text="Resultado de la Detección", font=("Helvetica", 12)).pack(pady=(0, 5))
    processed_panel = tk.Label(right_frame, relief="sunken", borderwidth=2, bg="gray90")
    processed_panel.pack(fill=tk.BOTH, expand=True)
    
    tk.Label(right_frame, text="Área de cada Segmento", font=("Helvetica", 10, "bold")).pack(pady=(10, 2))
    results_text = scrolledtext.ScrolledText(right_frame, height=8, font=("Courier New", 10), state='disabled', relief="sunken", borderwidth=2)
    results_text.pack(fill=tk.X, expand=False)
    
    # --- Controles Inferiores ---
    control_frame = tk.Frame(root)
    control_frame.pack(pady=10, fill=tk.X)

    calibration_frame = tk.Frame(control_frame)
    calibration_frame.pack(pady=5)
    tk.Label(calibration_frame, text="Distancia de Referencia (cm):", font=("Helvetica", 10)).pack(side=tk.LEFT, padx=(0, 5))
    calibration_var = tk.StringVar(value="1.0")
    tk.Entry(calibration_frame, textvariable=calibration_var, font=("Helvetica", 10), width=8, justify='center').pack(side=tk.LEFT)
    tk.Button(calibration_frame, text="Calibrar con 2 Clics", command=start_calibration, font=("Helvetica", 10)).pack(side=tk.LEFT, padx=10)

    button_frame = tk.Frame(control_frame)
    button_frame.pack(pady=5)
    tk.Button(button_frame, text="Cargar Imagen", command=open_image, font=("Helvetica", 10), width=18).pack(side=tk.LEFT, padx=5)
    tk.Button(button_frame, text="Procesar y Calcular Área", command=process_image, font=("Helvetica", 10, "bold"), width=22).pack(side=tk.LEFT, padx=5)
    
    # --- NUEVOS BOTONES DE EXPORTACIÓN ---
    add_to_list_button = tk.Button(button_frame, text="Añadir a Lista", command=add_to_export_list, font=("Helvetica", 10), width=15, state='disabled')
    add_to_list_button.pack(side=tk.LEFT, padx=5)
    export_button = tk.Button(button_frame, text="Exportar Lista", command=export_results, font=("Helvetica", 10, "bold"), bg="#D5E8D4", width=15, state='disabled')
    export_button.pack(side=tk.LEFT, padx=5)
    
    info_label = tk.Label(root, text="Carga una imagen para empezar.", font=("Helvetica", 10), fg="blue")
    info_label.pack(pady=5)
    area_label = tk.Label(root, text="Área Total Estimada: N/A", font=("Helvetica", 14, "bold"))
    area_label.pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    if predictor:
        main()
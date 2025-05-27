from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
import os
from pathlib import Path
import json

app = Flask(__name__, template_folder="templates")

# Configuración
MODEL_PATH = "modelo_comidas.keras"
CSV_GENERAL_PATH = "platos_peruanos_con_categoria_general.csv"
CSV_CATEGORIAS_PATH = "data_con_categorias.csv"
IMG_SIZE = (224, 224)

# Cargar modelo general
def cargar_modelo(ruta):
    ruta_modelo = Path(ruta)
    if not ruta_modelo.exists():
        raise FileNotFoundError(f"❌ El modelo no existe: {ruta}")
    model = tf.keras.models.load_model(ruta_modelo)
    return model

model_general = cargar_modelo(MODEL_PATH)

# Leer CSVs
df_general = pd.read_csv(CSV_GENERAL_PATH)
df_categorias = pd.read_csv(CSV_CATEGORIAS_PATH)

# Limpiar y estandarizar columnas clave
df_general['nombre_plato'] = df_general['nombre_plato'].astype(str).str.strip().str.replace("_", " ").str.lower()
df_general['categoria_general'] = df_general['categoria_general'].astype(str).str.strip().str.replace("_", " ").str.lower()
df_categorias['categoria_general'] = df_categorias['categoria_general'].astype(str).str.strip().str.replace("_", " ").str.lower()
df_categorias['NOMBRE DE LAS PREPARACIONES'] = df_categorias['NOMBRE DE LAS PREPARACIONES'].astype(str).str.strip()

# Leer categorías generales desde JSON
with open('general_classes.json', 'r') as f:
    CATEGORIAS_GENERALES = json.load(f)

# Preprocesar imagen
def preparar_imagen(img_file):
    try:
        img = Image.open(img_file).convert("RGB")
    except UnidentifiedImageError:
        raise ValueError("❌ No se pudo leer la imagen. Sube un archivo válido (JPG, PNG, BMP, etc.).")
    img = img.resize(IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.keras.applications.efficientnet_v2.preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)

# Predicción y construcción de resultados
def predecir_plato(img_batch):
    pred_general = model_general.predict(img_batch)[0]
    idx_general = np.argmax(pred_general)
    categoria_detectada = CATEGORIAS_GENERALES[idx_general].strip().replace("_", " ").lower()

    resultado_final = {
        "categoria_detectada": categoria_detectada.title(),
        "categoria_general": "",
        "plato_general": {},
        "platos_especificos": []
    }

    # Paso 1: Buscar plato general y obtener CATEGORIA_GENERAL
    fila_general = df_general[df_general['nombre_plato'] == categoria_detectada]
    if not fila_general.empty:
        fila = fila_general.iloc[0]
        categoria_general_puente = fila['categoria_general']
        resultado_final["categoria_general"] = categoria_general_puente.title()

        resultado_final["plato_general"] = {
            "nombre": fila['nombre_plato'].title(),
            "nutricion": {
                "agua": fila.get('Prom_Agua', 'No disponible'),
                "energia": fila.get('Prom_Energa', 'No disponible'),
                "grasa": fila.get('Prom_Grasa', 'No disponible'),
                "proteinas": fila.get('Prom_Protenas', 'No disponible')
            }
        }
    else:
        categoria_general_puente = categoria_detectada  # fallback

    # Paso 2: Buscar platos específicos usando CATEGORIA_GENERAL
    platos_especificos = df_categorias[df_categorias['categoria_general'] == categoria_general_puente]

    for _, fila in platos_especificos.iterrows():
        resultado_final["platos_especificos"].append({
            "nombre_preparacion": fila['NOMBRE DE LAS PREPARACIONES'].title(),
            "nutricion": {
                "agua": fila.get('Prom_Agua', 'No disponible'),
                "energia": fila.get('Prom_Energa', 'No disponible'),
                "grasa": fila.get('Prom_Grasa', 'No disponible'),
                "proteinas": fila.get('Prom_Protenas', 'No disponible')
            }
        })

    return resultado_final

# Rutas Flask
@app.route("/", methods=["GET"])
def home():
    return "✅ API de clasificación jerárquica de comida peruana activa"

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No se envió una imagen"}), 400

    img_file = request.files["image"]

    try:
        img_batch = preparar_imagen(img_file)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    resultado = predecir_plato(img_batch)

    return jsonify(resultado)

@app.route("/form", methods=["GET", "POST"])
def form():
    if request.method == "POST":
        if "image" not in request.files:
            return "No se envió una imagen"

        file = request.files["image"]
        if file.filename == "":
            return "Archivo vacío"

        try:
            img_batch = preparar_imagen(file)
        except ValueError as e:
            return f"Error: {e}"

        resultado = predecir_plato(img_batch)

        return render_template("result.html", result=resultado)

    return render_template("form.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

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
CSV_PATH = "data_con_categorias.csv"
IMG_SIZE = (224, 224)

# Cargar modelo general
def cargar_modelo(ruta):
    ruta_modelo = Path(ruta)
    if not ruta_modelo.exists():
        raise FileNotFoundError(f"❌ El modelo no existe: {ruta}")
    model = tf.keras.models.load_model(ruta_modelo)
    print(f"✅ Modelo cargado: {ruta_modelo}")
    return model

model_general = cargar_modelo(MODEL_PATH)

# Leer CSV completo
df_datos = pd.read_csv(CSV_PATH)

# Obtener lista única de categorías generales
CATEGORIAS_GENERALES = df_datos['categoria_general'].unique().tolist()

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

# Predicción jerárquica mejorada
def predecir_plato(img_batch):
    pred_general = model_general.predict(img_batch)[0]
    idx_general = np.argmax(pred_general)
    categoria = CATEGORIAS_GENERALES[idx_general]

    resultado_final = {
        "categoria_general": categoria.replace("_", " ").title(),
        "top5_platos": []
    }

    model_path = f"saved_models/{categoria}_model.h5"
    classes_path = f"saved_models/{categoria}_classes.json"

    if os.path.exists(model_path) and os.path.exists(classes_path):
        model_especializado = cargar_modelo(model_path)
        with open(classes_path, "r") as f:
            platos_names = json.load(f)

        pred_plato = model_especializado.predict(img_batch)[0]
        top5_idx = pred_plato.argsort()[-5:][::-1]

        for idx in top5_idx:
            plato = platos_names[idx]
            confianza = float(pred_plato[idx])

            fila = df_datos[df_datos['CATEGORIA_ESTANDAR'].str.lower() == plato.lower()]
            if not fila.empty:
                datos = fila.iloc[0].drop(['CATEGORIA_ESTANDAR', 'NOMBRE DE LAS PREPARACIONES', 'categoria_general'], errors='ignore')
                nutricion = datos.to_dict()
            else:
                nutricion = "No disponible"

            resultado_final["top5_platos"].append({
                "plato": plato.replace("_", " ").title(),
                "confianza": round(confianza, 4),
                "nutricion": nutricion
            })
    else:
        resultado_final["top5_platos"].append({
            "plato": categoria.replace("_", " ").title(),
            "confianza": round(float(pred_general[idx_general]), 4),
            "nutricion": "No disponible"
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

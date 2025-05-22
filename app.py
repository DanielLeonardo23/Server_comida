from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import os

app = Flask(__name__)

# Carga modelo
model = tf.keras.models.load_model("modelo_comidas.keras")

# Tamaño esperado por el modelo
IMG_SIZE = (224, 224)

# Ruta al CSV con nombres de clases y datos nutricionales
CSV_PATH = os.path.join("data", "data.csv")

# Carga clases desde CSV
df = pd.read_csv(CSV_PATH)
df.columns = [col.strip() for col in df.columns]
class_names = df["CATEGORIA_ESTANDAR"].tolist()

@app.route("/", methods=["GET"])
def home():
    return "✅ API Clasificador de Comidas Peruanas"

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No se envió una imagen"}), 400

    img_file = request.files["image"]
    img = Image.open(img_file).convert("RGB")
    img = img.resize(IMG_SIZE)

    # Preprocesamiento
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.keras.applications.efficientnet_v2.preprocess_input(img_array)
    img_batch = np.expand_dims(img_array, axis=0)

    # Predicción
    prediction = model.predict(img_batch)[0]
    top_5_indices = prediction.argsort()[-5:][::-1]

    results = []
    for i in top_5_indices:
        class_id = class_names[i]
        confianza = float(prediction[i])
        resultados_csv = df[df["CATEGORIA_ESTANDAR"].str.lower() == class_id.lower()]

        nutricion = {}
        if not resultados_csv.empty:
            row = resultados_csv.iloc[0].drop(["CATEGORIA_ESTANDAR", "NOMBRE DE LAS PREPARACIONES"])
            nutricion = row.to_dict()

        results.append({
            "class": class_id.replace("_", " ").title(),
            "confidence": f"{confianza * 100:.2f}%",
            "nutrition": nutricion
        })

    return jsonify(results)

@app.route("/form", methods=["GET", "POST"])
def form():
    if request.method == "POST":
        if "image" not in request.files:
            return "No se envió una imagen"

        file = request.files["image"]
        if file.filename == "":
            return "Archivo vacío"

        img = Image.open(file).convert("RGB")
        img = img.resize(IMG_SIZE)

        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.keras.applications.efficientnet_v2.preprocess_input(img_array)
        img_batch = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_batch)[0]
        top_5_indices = prediction.argsort()[-5:][::-1]

        top_predictions = []
        for i in top_5_indices:
            class_id = class_names[i]
            confianza = float(prediction[i])
            resultados_csv = df[df["CATEGORIA_ESTANDAR"].str.lower() == class_id.lower()]

            nutricion = {}
            if not resultados_csv.empty:
                row = resultados_csv.iloc[0].drop(["CATEGORIA_ESTANDAR", "NOMBRE DE LAS PREPARACIONES"])
                nutricion = row.to_dict()

            top_predictions.append({
                "class": class_id.replace("_", " ").title(),
                "confidence": f"{confianza * 100:.2f}%",
                "nutrition": nutricion
            })

        return render_template("result.html", resultados=top_predictions)

    return render_template("form.html")

from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

model = tf.keras.models.load_model("modelo_comidas.keras")

IMG_SIZE = (224, 224)

CLASS_NAMES = [
    'aguadito_de_pollo', 'arroz_chaufa_de_pollo', 'choclo_con_queso',
    'papa_a_la_huancaina', 'pollo_a_la_brasa_con_papas_fritas',
    'pollo_broaster_con_papas_fritas', 'tallarines_rojos_con_pollo',
    'tallarines_saltados_con_pollo', 'tallarines_verdes_con_bistec',
    'aji_de_gallina_peruano', 'anticuchos_peruanos',
    'arroz_con_pollo_peruano', 'carapulcra', 'causa_limena',
    'ceviche_peruano', 'juane', 'lomo_saltado_peruano', 'pachamanca',
    'sopa_seca_chinchana'
]

@app.route("/", methods=["GET"])
def home():
    return "API de clasificación de comida peruana"

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No se envió una imagen"}), 400

    img_file = request.files["image"]
    img = Image.open(img_file).convert("RGB")
    img = img.resize(IMG_SIZE)

    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.keras.applications.efficientnet_v2.preprocess_input(img_array)
    img_batch = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_batch)
    predicted_index = np.argmax(prediction)
    confidence = float(prediction[0][predicted_index])
    predicted_class = CLASS_NAMES[predicted_index].replace("_", " ").title()

    return jsonify({
        "predicted_class": predicted_class,
        "confidence": confidence
    })

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

        prediction = model.predict(img_batch)
        predicted_index = np.argmax(prediction)
        confidence = float(prediction[0][predicted_index])
        predicted_class = CLASS_NAMES[predicted_index].replace("_", " ").title()

        return render_template("result.html", pred=predicted_class, conf=f"{confidence*100:.2f}")

    return render_template("form.html")

# Cambiado para entorno de producción (Railway, Docker, etc.)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

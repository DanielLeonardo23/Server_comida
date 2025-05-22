from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import os
from pathlib import Path  # ✅ ← ESTA ES LA QUE FALTABA
app = Flask(__name__, template_folder="templates")


# Ruta del modelo (puede ser .keras, .h5 o carpeta SavedModel)
MODEL_PATH = "modelo_comidas.keras"  # Puedes cambiar esto si guardas en .h5 o carpeta

# Función para cargar el modelo de forma flexible
def cargar_modelo(ruta):
    ruta_modelo = Path(ruta)
    if not ruta_modelo.exists():
        raise FileNotFoundError(f"❌ El modelo no existe: {ruta}")

    try:
        # Si es archivo .keras o .h5
        model = tf.keras.models.load_model(ruta_modelo)
        print(f"✅ Modelo cargado desde archivo: {ruta_modelo}")
    except Exception as e:
        # Si es una carpeta SavedModel
        if ruta_modelo.is_dir():
            model = tf.keras.models.load_model(str(ruta_modelo))
            print(f"✅ Modelo cargado desde carpeta SavedModel: {ruta_modelo}")
        else:
            raise RuntimeError(f"❌ No se pudo cargar el modelo desde {ruta_modelo}\n{e}")
    return model

# Cargar modelo al iniciar
model = cargar_modelo(MODEL_PATH)


# Tamaño de imagen esperado
IMG_SIZE = (224, 224)

# Lista de clases (agrega todas las clases necesarias aquí)
CLASS_NAMES = [
    "adobo_de_cerdo_con_menestra_y_arroz",
    "adobo_de_cerdo_con_pure_y_arroz",
    "adobo_de_cerdo_y_arroz",
    "adobo_de_pollo_con_ensalada_y_arroz",
    "adobo_de_pollo_y_arroz",
    "adobo_de_res_con_arroz_y_pure",
    "adobo_de_res_con_menestra_y_arroz",
    "aeropuerto_de_pollo",
    "aeropuerto_tacon_pa",
    "aji_de_pollo_y_arroz",
    "albondigas_en_salsa_y_arroz",
    "alita_broaster_con_papas_fritas",
    "anticucho_con_papa_y_arroz",
    "arroz_a_la_jardinera_con_pollo",
    "arroz_arabe_con_pollo",
    "arroz_blanco_con_filete_de_pollo",
    "arroz_chaufa_con_pollo_chijaukay",
    "arroz_chaufa_con_pollo_tamarindo",
    "arroz_chaufa_con_pollo_taypa_especial",
    "arroz_chaufa_de_cerdo",
    "arroz_chaufa_de_mariscos",
    "arroz_chaufa_de_pollo",
    "arroz_chaufa_de_res",
    "arroz_chaufa_especial",
    "arroz_con_carne",
    "arroz_con_cerdo_con_sarsa_criolla",
    "arroz_con_frejol_y_seco_de_res",
    "arroz_con_frejoles_y_seco_de_pollo",
    "arroz_con_lentejas_con_churrasco",
    "arroz_con_mariscos",
    "arroz_con_pato",
    "arroz_con_pollo",
    "arroz_tapado",
    "asado_de_cerdo_con_arvejita_verde",
    "asado_de_cerdo_con_menestra_y_arroz",
    "asado_de_pollo_con_menestra_y_arroz",
    "asado_de_pollo_con_pure_y_arroz",
    "asado_de_pollo_y_arroz",
    "asado_de_res_con_menestra_y_arroz",
    "asado_de_res_con_pure_y_arroz",
    "asado_de_res_y_arroz",
    "bisteck_de_res_a_lo_pobre_con_arroz",
    "bisteck_de_res_con_arroz",
    "bisteck_de_res_con_papas_fritas",
    "cabrito_a_la_nortena_con_arroz",
    "cabrito_con_frejoles_y_arroz",
    "caldo_de_pollo",
    "caldo_de_res",
    "carapulcra_de_cerdo_y_arroz",
    "carapulcra_y_arroz",
    "carne_de_res_con_ensalada_y_arroz",
    "cau_cau_con_arroz",
    "causa_con_pollo_al_horno_y_arroz",
    "causa_criolla",
    "causa_de_palta",
    "causa_de_pescado",
    "causa_de_pollo",
    "causa_de_pulpa_de_cangrejo",
    "causa_rellena",
    "cebiche_con_chicharron",
    "cebiche_de_pescado",
    "cebiche_de_pescado_con_chicharron",
    "cebiche_de_pescado_con_pota",
    "cebiche_de_pollo_con_arroz",
    "cebiche_de_pota",
    "cebiche_mixto",
    "cebiche_y_chilcano_de_pescado",
    "cerdo_al_horno_con_arroz",
    "chanfainita_con_arroz",
    "chanfainita_con_tallarin",
    "chicharron_con_salsa_criolla_y_arroz",
    "chicharron_de_pescado",
    "chicharron_de_pescado_con_arroz",
    "chicharron_de_pescado_con_ensalada",
    "chicharron_de_pescado_con_frejoles",
    "chicharron_de_pescado_con_yuca",
    "chicharron_de_pollo",
    "chicharron_de_pollo_con_ensalada",
    "chicharron_de_pota",
    "chicharron_mixto",
    "chicharron_mixto_con_sarsa_criolla",
    "chicharron_mixto_con_yuca_frita",
    "chicken_quesadilla",
    "chilcano_de_pescado",
    "choclo_a_la_huancaina",
    "choclo_con_queso",
    "choritos_a_la_chalaca",
    "chuleta_a_la_parrilla_con_arroz",
    "chuleta_de_cerdo_con_papas_fritas",
    "chupe_de_camaron",
    "chupe_de_cangrejo",
    "chupe_de_chochoca",
    "chupe_de_choro",
    "chupe_de_habas",
    "chupe_de_mariscos",
    "chupe_de_pescado",
    "chupe_de_pollo",
    "chupe_de_trigo",
    "chupe_mixto",
    "churrasco_con_arroz",
    "churrasco_con_frejoles_y_arroz",
    "combinado_de_pollo",
    "combinado_de_res",
    "combinado_de_tallarin_rojo_con_papa",
    "consome_de_pollo",
    "cordero_a_la_nortena_con_frejoles",
    "costillas_barbiquiu_con_arroz",
    "costillas_con_papa_y_arroz",
    "crema_de_arveja",
    "crema_de_ocopa",
    "crema_de_pimiento",
    "crema_de_rocoto",
    "crema_de_verdura",
    "crema_de_zapallo",
    "croquetas",
    "ensalada",
    "ensalada_cesar",
    "ensalada_de_atun",
    "ensalada_de_cocona",
    "ensalada_de_fideos",
    "ensalada_de_lechuga",
    "ensalada_de_verduras",
    "ensalada_rusa",
    "ensalada_rusa_con_pollo_al_horno",
    "escabeche_de_pescado_con_arroz",
    "escabeche_de_pollo_con_arroz",
    "espesado_de_res_con_arroz",
    "estofado_de_lengua_con_arroz",
    "estofado_de_pavita_con_arroz",
    "estofado_de_pollo_con_arroz",
    "estofado_de_pollo_con_menestra",
    "estofado_de_res_con_arroz",
    "estofado_de_res_con_menestra_y_arroz",
    "fetuccini_a_lo_alfredo",
    "filete_de_pescado_con_arroz",
    "filete_de_pescado_con_menestra",
    "filete_de_pescado_con_papas_fritas",
    "filete_de_pescado_en_salsa",
    "filete_de_pollo_a_la_parrilla",
    "filete_de_pollo_con_papas_fritas",
    "flauta_rellena",
    "frejoles_con_asado_de_carne_y_arroz",
    "frejoles_con_bisteck_y_arroz",
    "frejoles_con_cabrito_a_la_nortena",
    "frejoles_con_pescado_y_arroz",
    "frejoles_con_pollo_a_la_nortena",
    "frejoles_con_pollo_al_vino",
    "frejoles_con_seco_a_la_nortena",
    "frejoles_con_seco_y_arroz",
    "garbanzo_con_arroz_y_chuleta_frita",
    "garbanzo_con_chuleta_y_arroz",
    "garbanzo_con_churrasco_ensalada",
    "garbanzo_con_pescado_y_arroz"
]


# Cargar CSV de nutrición
CSV_PATH = os.path.join(os.path.dirname(__file__), "data.csv")
df_nutricion = pd.read_csv(CSV_PATH) if os.path.exists(CSV_PATH) else None

@app.route("/", methods=["GET"])
def home():
    return "API de clasificación de comida peruana con nutrición"

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
    top_indices = prediction[0].argsort()[-5:][::-1]

    resultados = []
    for idx in top_indices:
        clase = CLASS_NAMES[idx]
        confianza = float(prediction[0][idx])
        plato = clase.replace("_", " ").title()

        nutricion = {}
        if df_nutricion is not None:
            fila = df_nutricion[df_nutricion["CATEGORIA_ESTANDAR"].str.lower() == clase.lower()]
            if not fila.empty:
                datos = fila.iloc[0].drop(['CATEGORIA_ESTANDAR', 'NOMBRE DE LAS PREPARACIONES'], errors='ignore')
                nutricion = datos.to_dict()

        resultados.append({
            "class": plato,
            "confidence": round(confianza, 4),
            "nutrition": nutricion or "No disponible"
        })

    return jsonify(resultados)

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
        top_indices = prediction[0].argsort()[-5:][::-1]

        resultados = []
        for idx in top_indices:
            clase = CLASS_NAMES[idx]
            confianza = float(prediction[0][idx])
            plato = clase.replace("_", " ").title()

            nutricion = {}
            if df_nutricion is not None:
                fila = df_nutricion[df_nutricion["CATEGORIA_ESTANDAR"].str.lower() == clase.lower()]
                if not fila.empty:
                    datos = fila.iloc[0].drop(['CATEGORIA_ESTANDAR', 'NOMBRE DE LAS PREPARACIONES'], errors='ignore')
                    nutricion = datos.to_dict()

            resultados.append({
                "class": plato,
                "confidence": f"{confianza * 100:.2f}",
                "nutrition": nutricion
            })

        return render_template("result.html", results=resultados)

    return render_template("form.html")


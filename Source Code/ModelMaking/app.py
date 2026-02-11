from flask import Flask, render_template, request, jsonify
import os
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

MODEL_PATH = "efficientnetv2_forgery.h5"


def load_or_download_model():
    """
    Loads the model if it exists locally.
    Otherwise downloads EfficientNetV2B0 (pretrained) and saves it locally.
    """
    if os.path.exists(MODEL_PATH):
        print("✅ Loading model from local storage...")
        model = tf.keras.models.load_model(MODEL_PATH)
    else:
        print("⬇️ Downloading EfficientNetV2B0 from TensorFlow...")
        base_model = EfficientNetV2B0(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

        # Add custom binary classification head
        x = GlobalAveragePooling2D()(base_model.output)
        x = Dense(128, activation="relu")(x)
        x = Dropout(0.4)(x)
        output = Dense(1, activation="sigmoid")(x)
        model = Model(inputs=base_model.input, outputs=output)

        # Freeze the base model
        for layer in base_model.layers:
            layer.trainable = False

        # Compile the model for inference
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

        # Save locally
        model.save(MODEL_PATH)
        print(f"💾 Model saved locally as {MODEL_PATH}")

    return model


# Load model at app startup
model = load_or_download_model()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict whether the uploaded image is Authentic or Forged.
    """
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    img_file = request.files["image"]
    img = Image.open(io.BytesIO(img_file.read())).convert("RGB")
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Make prediction
    pred = model.predict(img_array)[0][0]
    label = "Forged Image" if pred > 0.5 else "Authentic Image"

    return jsonify({"prediction": label, "confidence": round(float(pred), 4)})


if __name__ == "__main__":
    app.run(debug=True)
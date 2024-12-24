import gradio as gr
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Fungsi untuk pemrosesan gambar dan prediksi
def process_image(input_image):
    # Memuat model yang sudah dilatih
    model = load_model("final_model.keras")

    # Ubah gambar input ke format yang cocok
    input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
    
    # Lakukan contrast stretching pada gambar
    min_val = np.min(input_image)
    max_val = np.max(input_image)
    processed_image = (input_image - min_val) * (255 / (max_val - min_val))
    processed_image = processed_image.astype(np.uint8)

    # Resize gambar ke ukuran yang sesuai dengan model (224x224)
    resized_image = cv2.resize(processed_image, (224, 224))  # Ubah ke 224x224
    img_array = np.expand_dims(resized_image, axis=0) / 255.0

    # Prediksi menggunakan model yang sudah dilatih
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    
    # Daftar nama kelas yang baru
    class_names = ['bicycle', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cup', 'dog', 'motorbike', 'people', 'table']
    predicted_label = class_names[predicted_class]
    
    # Skor prediksi dalam persen
    prediction_score = np.max(predictions) * 100

    return input_image, processed_image, predicted_label, f"Confidence: {prediction_score:.2f}%"

# UI menggunakan Gradio
iface = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="numpy", image_mode='RGB', label="Upload Image"),
    outputs=[
        gr.Image(type="numpy", label="Original Image"),
        gr.Image(type="numpy", label="Processed Image"),
        gr.Textbox(label="Predicted Class"),
        gr.Textbox(label="Prediction Confidence")
    ],
    title="Target Identification with Contrast Stretching",
    description="Upload a dark image to process and classify it using the pre-trained model."
)

# Menjalankan antarmuka
iface.launch()

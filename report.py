import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Path dataset yang digunakan
dataset_path = "C:/Alvin's File/Tugas/Tugas TelU/Semester 7/PCD/TUGAS AKHIR/FINAL REPORT/SOURCE CODE/ExDark"

# Data Preprocessing dan Augmentasi untuk data validasi
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

# Membaca data validasi dari folder
validation_generator = test_datagen.flow_from_directory(
    os.path.join(dataset_path, 'validation'),
    target_size=(224, 224),  
    batch_size=32,
    class_mode='categorical', 
    shuffle=False  
)

# Memuat model yang sudah dilatih
model = load_model('final_model.keras')

# Mengambil label yang benar (ground truth)
y_true = validation_generator.classes

# Melakukan prediksi pada data validasi
steps = int(np.ceil(validation_generator.samples / validation_generator.batch_size)) 
y_pred = model.predict(validation_generator, steps=steps, verbose=1)
y_pred = np.argmax(y_pred, axis=1)  

# Menghitung metrik evaluasi
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='macro')  
recall = recall_score(y_true, y_pred, average='macro')
f1 = f1_score(y_true, y_pred, average='macro')

# Menampilkan hasil metrik
print(f"Akurasi: {accuracy:.4f}")
print(f"Presisi: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")


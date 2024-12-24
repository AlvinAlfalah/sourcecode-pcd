import os
import numpy as np
import cv2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Path dataset yang digunakan
dataset_path = "C:/Alvin's File/Tugas/Tugas TelU/Semester 7/PCD/TUGAS AKHIR/FINAL REPORT/SOURCE CODE/ExDark"

# Data Augmentation dan Preprocessing
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,  # Normalisasi gambar
    rotation_range=20,  # Rotasi gambar secara acak
    width_shift_range=0.2,  # Geser gambar secara horizontal
    height_shift_range=0.2,  # Geser gambar secara vertikal
    shear_range=0.2,  # Geser sudut gambar
    zoom_range=0.2,  # Zoom gambar secara acak
    horizontal_flip=True,  # Pembalikan gambar secara horizontal
    fill_mode='nearest'  # Pengisian ulang piksel yang hilang setelah transformasi
)

test_datagen = ImageDataGenerator(rescale=1.0/255.0)

# Membaca data dari folder
train_generator = train_datagen.flow_from_directory(
    os.path.join(dataset_path, 'train'),
    target_size=(224, 224),  # Ukuran input yang lebih besar untuk MobileNetV2
    batch_size=32,
    class_mode='categorical'
)

validation_generator = test_datagen.flow_from_directory(
    os.path.join(dataset_path, 'validation'),
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Memuat model pre-trained MobileNetV2 tanpa top layer
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Menambahkan layer custom (Fully Connected Layer) di atas base model
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Menambahkan GlobalAveragePooling2D layer
x = Dense(128, activation='relu')(x)  # Fully connected layer
predictions = Dense(train_generator.num_classes, activation='softmax')(x)  # Output layer

# Membangun model final
model = Model(inputs=base_model.input, outputs=predictions)

# Freezing base model layers
for layer in base_model.layers:
    layer.trainable = False

# Kompilasi model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Menyimpan model dengan model checkpoint
checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)

# Menyusun langkah per epoch dan validasi berdasarkan ukuran batch
steps_per_epoch = train_generator.samples // train_generator.batch_size
validation_steps = validation_generator.samples // validation_generator.batch_size

# Melatih model dengan jumlah epoch yang lebih sedikit
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,  # Sesuaikan dengan jumlah dataset
    epochs=10,  # Mengurangi jumlah epoch untuk mempercepat
    validation_data=validation_generator,
    validation_steps=validation_steps,  # Sesuaikan dengan jumlah dataset
    callbacks=[checkpoint]
)

# Menyimpan model yang telah dilatih
model.save('final_model.keras')

# Evaluasi model pada data validasi
# Mengambil label dan prediksi untuk evaluasi metrik
y_true = validation_generator.classes
y_pred = model.predict(validation_generator, steps=validation_steps, verbose=1)
y_pred = np.argmax(y_pred, axis=1)  # Mengambil kelas prediksi dengan argmax

# Menghitung metrik evaluasi
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='macro')  # 'macro' untuk multi-kelas
recall = recall_score(y_true, y_pred, average='macro')
f1 = f1_score(y_true, y_pred, average='macro')

# Tampilkan hasil metrik
print(f"Akurasi: {accuracy:.4f}")
print(f"Presisi: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

```
# Deep Learning Image Classification: Pepper Bell Disease Detection

## Deskripsi Proyek

Proyek ini bertujuan untuk membangun model klasifikasi citra berbasis Convolutional Neural Network (CNN) untuk mendeteksi penyakit pada tanaman lada (Pepper Bell). Dataset terdiri dari dua kelas utama: daun yang sehat (`Pepper__bell___healthy`) dan daun yang terinfeksi penyakit bakteri spot (`Pepper__bell___Bacterial_spot`). Model yang dibangun menggunakan TensorFlow/Keras dan disimpan dalam format SavedModel serta dioptimasi untuk deployment dengan TensorFlow Serving dan TensorFlow.js.

Proyek ini juga mencakup tahap inference pada dataset gambar baru yang disimpan di Google Drive, dengan proses evaluasi prediksi yang menghasilkan laporan distribusi prediksi per kelas.

---

## Struktur Direktori

```
├── saved\_model/
├── inference/
├── main.py
├── utils/
├── tests/
├── README.md

````

---

## Library dan Paket yang Digunakan

- Python 3.x
- TensorFlow 2.x
- Keras
- numpy
- pandas
- matplotlib
- seaborn
- PIL
- scikit-learn
- tensorflowjs
- google.colab (untuk mount Drive)
- IPython.display

Instalasi utama bisa dilakukan dengan:

```bash
pip install tensorflow tensorflowjs pandas matplotlib seaborn scikit-learn pillow
````

---

## Langkah-Langkah Utama Proyek

### 1. Training Model CNN

* Dataset diload dan diproses dengan augmentasi.
* Model CNN dibangun menggunakan beberapa Conv2D, MaxPooling2D, Flatten, Dense, dan Dropout layers.
* Model dilatih hingga mencapai akurasi > 85% pada data validasi.
* Model disimpan dalam format TensorFlow SavedModel.

### 2. Deployment Model

* Model siap digunakan untuk inference dengan memanfaatkan TensorFlow Serving melalui layer `TFSMLayer`.
* Model juga dikonversi ke format TensorFlow\.js dan TensorFlow Lite (opsional).

### 3. Inference dan Evaluasi

* Gambar dari folder Google Drive (inference dataset) diambil secara acak (150 sampel).
* Dilakukan prediksi kelas menggunakan model yang sudah disimpan.
* Hasil prediksi ditampilkan secara visual dalam grid dengan nama file, kelas prediksi, dan confidence.
* Ringkasan distribusi prediksi per kelas disajikan dalam bentuk tabel dan grafik.
* Karena dataset inference tidak memiliki label ground truth yang pasti, evaluasi fokus pada distribusi prediksi tanpa menghitung akurasi.

---

## Contoh Kode Utama Inference dan Evaluasi

```python
from tensorflow.keras.preprocessing import image
from keras.layers import TFSMLayer
import tensorflow as tf
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython.display import display, HTML

# Load model SavedModel
saved_model_path = 'saved_model/my_model'
model = tf.keras.Sequential([TFSMLayer(saved_model_path, call_endpoint='serving_default')])

# Folder inference
inference_dir = '/content/drive/MyDrive/Collage/DBS Coding Camp/Submission/Deep Learning/Dataset/Inference'
class_names = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy']

def predict_and_evaluate(directory, n_samples=150, grid_rows=3, grid_cols=5):
    # Sampling dan prediksi gambar
    ...
    # Visualisasi dan laporan distribusi prediksi
    ...
    return df_results
```

---

## Hasil dan Kesimpulan

* Model CNN mampu melakukan prediksi klasifikasi penyakit pada daun lada dengan hasil inference yang memuaskan.
* Distribusi prediksi menunjukkan model mendeteksi kedua kelas dengan proporsi yang logis.
* Confidence prediksi cukup tinggi, menandakan model memiliki keyakinan terhadap hasilnya.
* Evaluasi pada dataset tanpa label ground truth fokus pada distribusi dan visualisasi hasil.
* Proyek ini membuktikan pipeline end-to-end mulai dari training, penyimpanan model, hingga deployment dan inference berjalan dengan lancar.

---

## Penggunaan

1. Pastikan semua library telah diinstall.
2. Mount Google Drive jika menggunakan Google Colab untuk mengakses dataset inference.
3. Jalankan script `main.py` atau notebook untuk melakukan training dan inference.
4. Lihat hasil visualisasi dan laporan yang dihasilkan di output.

---

## Referensi

* [TensorFlow Official Documentation](https://www.tensorflow.org/)
* [Keras API Guide](https://keras.io/)
* [TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving)
* Dataset PlantVillage (pepper bell disease)

**Terima kasih**
*Ananta Boemi Adji*